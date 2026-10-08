"""Symmetric dual-encoder group-head MLP for sky-coefficient transfer.

Maps the two sky arms' decomposition coefficients plus per-arm observing
context onto the science arm's coefficients.  One class,
``DualEncoderGroupHeadMLPCompressed``; every construction argument below is
live and exercised by the deployed configuration.

Forward path
------------
1.  ``sky_encoder`` -- one shared MLP applied to each arm's ``(score, ctx)``
    concatenation, so the two arms are treated symmetrically.
2.  The two arm embeddings are combined as ``(mean, diff, |diff|)``.  ``diff``
    carries the near/far disagreement, which is the only direct evidence of how
    much the sky varies across the field; ``|diff|`` lets the trunk use its
    magnitude without its sign.
3.  ``ctx_encoder`` embeds the science-arm context; ``trunk`` fuses all four.
4.  Six per-group heads (moon, zodi, continuum, mesospheric, ionospheric,
    atomic) emit signed score-space predictions from the trunk output.
5.  Each group's prediction is added to a blend of the two arms' own scores,
    ``alpha * near + (1 - alpha) * far``.  For ``moon``, ``zodi`` and
    ``continuum`` alpha is per-row, ``sigmoid`` of a linear function of three
    context features; the other three groups use a learned scalar.
6.  ``zodi`` and ``continuum`` additionally route through isolated branches fed
    only physically-motivated context subsets, keeping their predictions from
    depending on the full context vector.
7.  ``moon`` and ``zodi`` are coupled: a shared branch over the moon-scatter
    plus zodi-geometry context union produces a latent, projected additively
    into both heads by zero-initialised linear maps.  The coupling supplies the
    moon head -- which sits on the shared trunk -- with the restricted context
    it would otherwise never see, and is worth ~4.5 percentage points of the
    moon's accuracy gain over copying the near arm.
"""

from __future__ import annotations

from collections.abc import Sequence, Mapping

import numpy as np
import torch
import torch.nn as nn


def _make_mlp(in_dim: int, dims: Sequence[int]) -> nn.Module:
    layers: list[nn.Module] = []
    d0 = int(in_dim)
    for d1 in dims:
        d1 = int(d1)
        layers += [nn.Linear(d0, d1), nn.LayerNorm(d1), nn.GELU()]
        d0 = d1
    return nn.Sequential(*layers) if layers else nn.Identity()


def _make_shared_head(in_dim: int, head_dim: int, n_out: int) -> nn.Module:
    """trunk_out -> head_dim -> n_out with a single GELU between."""
    return nn.Sequential(
        nn.Linear(int(in_dim), int(head_dim)),
        nn.GELU(),
        nn.Linear(int(head_dim), int(n_out)),
    )


def _make_isolated_head(in_dim: int, extra_dims: Sequence[int], n_out: int) -> nn.Module:
    """hidden_b -> [extra_dims...] -> n_out with GELU between (empty extras = plain Linear)."""
    extras = tuple(int(v) for v in extra_dims)
    if not extras:
        return nn.Linear(int(in_dim), int(n_out))
    layers: list[nn.Module] = []
    d0 = int(in_dim)
    for d1 in extras:
        layers += [nn.Linear(d0, d1), nn.GELU()]
        d0 = d1
    layers += [nn.Linear(d0, int(n_out))]
    return nn.Sequential(*layers)


def _resolve_ctx_indices(ctx_names, want, purpose):
    """Return the ordered list of ctx indices whose lowercased name is in ``want``.

    Raises if none match — the deployed pipeline always ships the requested
    features; a mismatch means the ctx wiring is wrong upstream.
    """
    ctx_names_lower = [str(n).strip().lower() for n in ctx_names]
    want_lower = [str(x).strip().lower() for x in want]
    idx = [i for i, n in enumerate(ctx_names_lower) if n in want_lower]
    missing = [w for w in want_lower if w not in ctx_names_lower]
    if len(idx) == 0:
        raise ValueError(
            f"{purpose}: none of the requested features {want_lower!r} are "
            f"in ctx_names."
        )
    if missing:
        print(f"{purpose}: missing features {missing!r} (they will be skipped).")
    return idx, tuple(ctx_names_lower[i] for i in idx)


class DualEncoderGroupHeadMLPCompressed(nn.Module):
    """Deployed compressed dual-encoder group-head MLP.

    Architecture (matches the state_dict layout of
    ``mlp_ensemble_split_zodi_current.pt``):

    1. Shared ``sky_encoder`` MLP encodes each arm's ``(score, ctx)`` concat.
    2. Arm embeddings combined as ``(mean, diff, |diff|)``; a ``ctx_encoder``
       encodes the science-arm ctx.  All four are fused by a ``trunk`` MLP.
    3. Six per-group heads (moon, zodi, continuum, mesospheric, ionospheric,
       atomic) emit SIGNED score-space predictions.
    4. Blending: per-group ``blend_alpha_direct`` scalar in [eps, 1-eps]
       combines the two arm scores; for ``('moon','zodi','continuum')`` the
       scalar is replaced by a per-row ctx-dependent alpha predictor.
    5. Zodi and continuum groups additionally route through isolated branches
       keyed on physically-motivated ctx feature subsets.
    6. Moon+zodi are additively coupled via a small shared ``moon_zodi``
       branch whose latent is projected (zero-init) into both heads.
    """

    # 2026-08-26 Phase A: which groups get context-dependent alpha.
    _ALPHA_CTX_GROUPS: tuple[str, ...] = ("moon", "zodi", "continuum")

    def __init__(
        self,
        *,
        n_score: int,
        n_ctx: int,
        group_score_dims: dict[str, int],
        ctx_names: Sequence[str],
        encoder_dims: tuple[int, ...] = (768, 384),
        ctx_dims: tuple[int, ...] = (96,),
        trunk_dims: tuple[int, ...] = (320, 160),
        head_dim: int = 192,
        zodi_head_extra_dims: tuple[int, ...] = (32,),
        continuum_head_extra_dims: tuple[int, ...] = (64,),
        continuum_branch_dims: tuple[int, ...] = (128, 64),
        blend_init_alpha: float | Mapping[str, float] = 0.7,
        alpha_ctx_features: Sequence[str] = ("moon_up_smooth", "ecl_beta_deg", "airmass"),
        zodi_ctx_restriction: Sequence[str] = (),
        continuum_ctx_restriction: Sequence[str] = (),
        moon_zodi_ctx_restriction: Sequence[str] = (),
        moon_zodi_coupling_dims: tuple[int, ...] = (64, 32),
    ) -> None:
        super().__init__()
        if not zodi_ctx_restriction:
            raise ValueError("zodi_ctx_restriction must be non-empty for the deployed split-zodi model.")
        if not continuum_ctx_restriction:
            raise ValueError("continuum_ctx_restriction must be non-empty for the deployed model.")
        if not moon_zodi_ctx_restriction:
            raise ValueError("moon_zodi_ctx_restriction must be non-empty for the deployed model.")
        if not alpha_ctx_features:
            raise ValueError("alpha_ctx_features must be non-empty for the deployed model.")

        self.ctx_names = tuple(str(n).strip().lower() for n in ctx_names)
        self.group_score_dims = dict(group_score_dims)

        # Cumulative offsets track how compress_coefs_to_scores concatenates the per-group score blocks.
        _offsets = np.cumsum([0] + [int(group_score_dims[g]) for g in group_score_dims])
        self._score_offsets = {
            g: (int(_offsets[i]), int(_offsets[i + 1]))
            for i, g in enumerate(group_score_dims)
        }

        # ---- Encoders + trunk -------------------------------------------------
        self.sky_encoder = _make_mlp(int(n_score) + int(n_ctx), encoder_dims)
        sky_embed_dim = int(encoder_dims[-1])

        self.ctx_encoder = _make_mlp(int(n_ctx), ctx_dims)
        ctx_embed_dim = int(ctx_dims[-1]) if len(ctx_dims) > 0 else int(n_ctx)

        fusion_in = 3 * sky_embed_dim + ctx_embed_dim
        self.trunk = _make_mlp(fusion_in, trunk_dims)
        trunk_out = int(trunk_dims[-1]) if len(trunk_dims) > 0 else fusion_in

        # ---- Shared-trunk heads (one per group) -------------------------------
        self.heads = nn.ModuleDict({
            g: _make_shared_head(trunk_out, int(head_dim), int(n_g))
            for g, n_g in group_score_dims.items()
        })

        # ---- Per-group learnable blend alpha (direct parametrization) --------
        #
        # `blend_init_alpha` is a scalar for every group, or a mapping giving
        # per-group values (missing groups fall back to the mapping's
        # 'default', else 0.7).  Per-group matters because 0.85 encodes "the
        # near arm is mostly right", which holds for moon/zodi but not for
        # mesospheric: the best naive baseline there is B2_mean_geo, the
        # EQUAL-weight two-arm mean, i.e. alpha = 0.5.  Starting a group at
        # the wrong end costs gradient it may not have -- mesospheric carries
        # 1.6% of the loss weight (m_g/sqrt(n_g) with n_g = 358), and in the
        # deployed run its alpha only reached 0.679 from 0.850 while every
        # other group stayed pinned at its init.
        self.blend_alpha_eps = 1e-3
        _clip = lambda a: float(np.clip(float(a), self.blend_alpha_eps,
                                        1 - self.blend_alpha_eps))
        if isinstance(blend_init_alpha, Mapping):
            _dflt = _clip(blend_init_alpha.get("default", 0.7))
            _init_by_group = {str(g): _clip(blend_init_alpha.get(g, _dflt))
                              for g in group_score_dims}
        else:
            _a = _clip(blend_init_alpha)
            _init_by_group = {str(g): _a for g in group_score_dims}
        self.blend_init_alpha_by_group = dict(_init_by_group)
        _logit = lambda a: float(np.log(a / (1.0 - a)))
        self.blend_alpha_direct = nn.ParameterDict({
            str(g): nn.Parameter(torch.tensor(_init_by_group[str(g)]))
            for g in group_score_dims
        })

        # ---- Context-dependent alpha for the anisotropic groups --------------
        idx, self.alpha_ctx_features = _resolve_ctx_indices(
            ctx_names, alpha_ctx_features, "alpha_ctx_features")
        self.register_buffer("alpha_ctx_idx", torch.tensor(idx, dtype=torch.long))
        _alpha_groups = tuple(g for g in self._ALPHA_CTX_GROUPS if g in group_score_dims)
        self.alpha_ctx_groups = _alpha_groups
        # alpha is a pure AMPLITUDE knob: it interpolates the two arms' scores
        # and cannot change their colour.  It reads the SCIENCE context only.
        # Feeding it the near/far contexts as well, and giving it depth, were
        # both measured (2026-09-04) and both made the moon tail WORSE -- and
        # the leverage measurement showed alpha already commands 1.3-1.5x the
        # amplitude range the tail error needs, so it was never short of reach.
        # See ablations.RETIRED for the numbers.
        self.alpha_predictors = nn.ModuleDict({
            g: nn.Linear(len(idx), 1) for g in _alpha_groups
        })
        # Zero the weights and set the bias so that sigmoid(bias) = the
        # group's own init_alpha at t=0.
        with torch.no_grad():
            for _g, _p in self.alpha_predictors.items():
                _p.weight.zero_()
                _p.bias.fill_(_logit(_init_by_group[str(_g)]))
        print(
            f"DualEncoderGroupHeadMLPCompressed: context-dependent alpha active, "
            f"features = {self.alpha_ctx_features}, groups = {self.alpha_ctx_groups}."
        )

        # ---- Isolated zodi branch --------------------------------------------
        z_idx, self.zodi_ctx_restriction = _resolve_ctx_indices(
            ctx_names, zodi_ctx_restriction, "zodi_ctx_restriction")
        self.register_buffer("zodi_ctx_idx", torch.tensor(z_idx, dtype=torch.long))
        n_zodi_score = int(group_score_dims["zodi"])
        zodi_in = 3 * len(z_idx) + 2 * n_zodi_score
        # 2026-08-26 Phase B/C: hidden dims (64, 32) matches saved checkpoint layout.
        _z_hidden_a, _z_hidden_b = 64, 32
        self.zodi_branch = nn.Sequential(
            nn.Linear(zodi_in, _z_hidden_a),
            nn.LayerNorm(_z_hidden_a), nn.GELU(),
            nn.Linear(_z_hidden_a, _z_hidden_b),
            nn.LayerNorm(_z_hidden_b), nn.GELU(),
        )
        self.zodi_head_isolated = _make_isolated_head(_z_hidden_b, zodi_head_extra_dims, n_zodi_score)
        print(
            f"DualEncoderGroupHeadMLPCompressed: isolated zodi branch active, "
            f"ctx features = {self.zodi_ctx_restriction}, n_score={n_zodi_score}, "
            f"head input dim={zodi_in}, zodi_head_extra_dims={tuple(zodi_head_extra_dims)}."
        )

        # ---- Isolated continuum branch ---------------------------------------
        c_idx, self.continuum_ctx_restriction = _resolve_ctx_indices(
            ctx_names, continuum_ctx_restriction, "continuum_ctx_restriction")
        self.register_buffer("continuum_ctx_idx", torch.tensor(c_idx, dtype=torch.long))
        n_c_score = int(group_score_dims["continuum"])
        c_in = 3 * len(c_idx) + 2 * n_c_score
        _cb_dims = tuple(int(v) for v in continuum_branch_dims)
        if len(_cb_dims) != 2:
            raise ValueError(f"continuum_branch_dims must be a 2-tuple; got {_cb_dims!r}.")
        _c_hidden_a, _c_hidden_b = _cb_dims
        self.continuum_branch = nn.Sequential(
            nn.Linear(c_in, _c_hidden_a),
            nn.LayerNorm(_c_hidden_a), nn.GELU(),
            nn.Linear(_c_hidden_a, _c_hidden_b),
            nn.LayerNorm(_c_hidden_b), nn.GELU(),
        )
        self.continuum_head_isolated = _make_isolated_head(
            _c_hidden_b, continuum_head_extra_dims, n_c_score)
        print(
            f"DualEncoderGroupHeadMLPCompressed: isolated continuum branch active, "
            f"ctx features = {self.continuum_ctx_restriction}, n_score={n_c_score}, "
            f"head input dim={c_in}, continuum_branch_dims={_cb_dims}, "
            f"continuum_head_extra_dims={tuple(continuum_head_extra_dims)}."
        )

        # ---- Additive moon-zodi coupling (Phase F) ---------------------------
        mz_idx, self.moon_zodi_ctx_restriction = _resolve_ctx_indices(
            ctx_names, moon_zodi_ctx_restriction, "moon_zodi_ctx_restriction")
        self.register_buffer("moon_zodi_ctx_idx", torch.tensor(mz_idx, dtype=torch.long))
        n_m = int(group_score_dims["moon"])
        n_z = int(group_score_dims["zodi"])
        mz_in = 3 * len(mz_idx) + 2 * (n_m + n_z)
        _mzc_dims = tuple(int(v) for v in moon_zodi_coupling_dims)
        if len(_mzc_dims) != 2:
            raise ValueError(f"moon_zodi_coupling_dims must be a 2-tuple; got {_mzc_dims!r}.")
        _mzc_hidden_a, _mzc_hidden_b = _mzc_dims
        self.moon_zodi_coupling_branch = nn.Sequential(
            nn.Linear(mz_in, _mzc_hidden_a),
            nn.LayerNorm(_mzc_hidden_a), nn.GELU(),
            nn.Linear(_mzc_hidden_a, _mzc_hidden_b),
            nn.LayerNorm(_mzc_hidden_b), nn.GELU(),
        )
        self.moon_zodi_moon_projector = nn.Linear(_mzc_hidden_b, n_m)
        self.moon_zodi_zodi_projector = nn.Linear(_mzc_hidden_b, n_z)
        # Zero-init projectors: at t=0 the additive contribution is exactly zero.
        with torch.no_grad():
            for _p in (self.moon_zodi_moon_projector, self.moon_zodi_zodi_projector):
                _p.weight.zero_()
                _p.bias.zero_()
        print(
            f"DualEncoderGroupHeadMLPCompressed: additive moon-zodi coupling active, "
            f"ctx features = {self.moon_zodi_ctx_restriction}, n_moon_score={n_m}, "
            f"n_zodi_score={n_z}, branch input dim={mz_in}, coupling_dims={_mzc_dims}, "
            f"projectors zero-init."
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        near_score: torch.Tensor,
        far_score: torch.Tensor,
        near_ctx: torch.Tensor,
        far_ctx: torch.Tensor,
        sci_ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        e_near = self.sky_encoder(torch.cat([near_score, near_ctx], dim=1))
        e_far = self.sky_encoder(torch.cat([far_score, far_ctx], dim=1))
        e_mean = 0.5 * (e_near + e_far)
        e_diff = e_near - e_far
        e_absdiff = torch.abs(e_diff)
        e_ctx = self.ctx_encoder(sci_ctx)

        h = self.trunk(torch.cat([e_mean, e_diff, e_absdiff, e_ctx], dim=1))

        # Additive moon-zodi latent (Phase F): computed once per batch.
        mzctx_sci = sci_ctx[:, self.moon_zodi_ctx_idx]
        mzctx_near = near_ctx[:, self.moon_zodi_ctx_idx]
        mzctx_far = far_ctx[:, self.moon_zodi_ctx_idx]
        lo_m, hi_m = self._score_offsets["moon"]
        lo_z, hi_z = self._score_offsets["zodi"]
        mzin = torch.cat([
            mzctx_sci, mzctx_near, mzctx_far,
            near_score[:, lo_m:hi_m], far_score[:, lo_m:hi_m],
            near_score[:, lo_z:hi_z], far_score[:, lo_z:hi_z],
        ], dim=1)
        mz_latent = self.moon_zodi_coupling_branch(mzin)

        alpha_ctx_sci = sci_ctx[:, self.alpha_ctx_idx]

        out: dict[str, torch.Tensor] = {}
        for g, head in self.heads.items():
            lo, hi = self._score_offsets[g]
            if g in self.alpha_predictors:
                alpha = torch.sigmoid(self.alpha_predictors[g](alpha_ctx_sci))
                blend = alpha * near_score[:, lo:hi] + (1.0 - alpha) * far_score[:, lo:hi]
            else:
                alpha = self.blend_alpha_direct[g]
                blend = alpha * near_score[:, lo:hi] + (1.0 - alpha) * far_score[:, lo:hi]

            if g == "zodi":
                zctx_sci = sci_ctx[:, self.zodi_ctx_idx]
                zctx_near = near_ctx[:, self.zodi_ctx_idx]
                zctx_far = far_ctx[:, self.zodi_ctx_idx]
                zin = torch.cat([
                    zctx_sci, zctx_near, zctx_far,
                    near_score[:, lo:hi], far_score[:, lo:hi],
                ], dim=1)
                z_out = self.zodi_head_isolated(self.zodi_branch(zin))
                z_out = z_out + self.moon_zodi_zodi_projector(mz_latent)
                out[g] = blend + z_out
            elif g == "continuum":
                cctx_sci = sci_ctx[:, self.continuum_ctx_idx]
                cctx_near = near_ctx[:, self.continuum_ctx_idx]
                cctx_far = far_ctx[:, self.continuum_ctx_idx]
                cin = torch.cat([
                    cctx_sci, cctx_near, cctx_far,
                    near_score[:, lo:hi], far_score[:, lo:hi],
                ], dim=1)
                out[g] = blend + self.continuum_head_isolated(self.continuum_branch(cin))
            elif g == "moon":
                head_out = head(h) + self.moon_zodi_moon_projector(mz_latent)
                out[g] = blend + head_out
            else:
                out[g] = blend + head(h)
        return out


__all__ = [
    "DualEncoderGroupHeadMLPCompressed",
]
