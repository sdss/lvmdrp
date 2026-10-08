"""Serialize / deserialize trained ensembles for inference.

The training pipeline in ``mlp_predictor.trainer`` produces an
``EnsembleArtifacts`` dataclass whose ``mlp_artifacts`` dict carries the
per-seed torch models, the RobustScaler instances for scores + ctx, the
per-group compressors, the Jensen post-training lift, and every downstream
knob the inference path needs.  This module writes those objects to a single
``.pt`` archive on disk and reads them back into a shape that is drop-in
compatible with :func:`mlp_predictor.trainer.predict_sci_coefficients_default`
and :func:`mlp_predictor.inference.predict_sky_from_minimal_inputs`.

Inference-necessary state is persisted, plus the per-member training
history (loss and blend-alpha per epoch, best epoch, best val loss) so a
saved run can be replotted and compared later without a retrain.  The
train/val/test ROW INDICES are still dropped: they are the only part that
would leak the split, and the notebook reproduces them deterministically
from ``(obstime_mjd, moon_phase, seed=42)`` anyway.
"""

from __future__ import annotations

from collections.abc import Mapping

from pathlib import Path
from typing import Mapping, Any

import torch

from .model import DualEncoderGroupHeadMLPCompressed

# v2 (2026-09-09) added the constraint-derived amplitude rules; a v1 file
# loads without them and silently mis-predicts, so v1 is refused.
FORMAT_VERSION = 2


def _cpu_state_dict(model: torch.nn.Module) -> dict:
    """Detach + move a model state_dict to CPU tensors for portable serialisation."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_ensemble(ensemble_artifacts: Mapping[str, Any], out_path: str | Path) -> Path:
    """Persist an ensemble artifact to a single ``.pt`` archive.

    Parameters
    ----------
    ensemble_artifacts:
        The ``mlp_artifacts`` dict produced by :class:`~mlp_predictor.trainer.Trainer`
        (equivalently ``EnsembleArtifacts.mlp_artifacts``).  Must have
        ``is_ensemble=True`` and a ``members`` list.
    out_path:
        Destination file path.  Parent directories are created automatically.

    Returns
    -------
    Path
        The absolute path to the written archive.
    """
    if not ensemble_artifacts.get("is_ensemble", False):
        raise ValueError(
            "save_ensemble expects an ensemble artifact "
            "(is_ensemble=True); got a single-seed artifact.")
    members = ensemble_artifacts["members"]
    if not members:
        raise ValueError("Ensemble has no members to save.")

    first = members[0]
    payload = {
        "format_version": FORMAT_VERSION,
        "config": dict(first["config"]),
        "seeds": list(ensemble_artifacts["seeds"]),
        "member_state_dicts": [_cpu_state_dict(m["model"]) for m in members],
        "score_scaler": first["score_scaler"],
        "ctx_scaler":   first["ctx_scaler"],
        "compressors":  ensemble_artifacts["compressors"],
        # LEGACY, kept so an older loader still works: member 0's calibration.
        "jensen_corrections": first["jensen_corrections"],
        # PER-MEMBER calibration (2026-09-12).  `jensen_corrections` is fitted
        # from each member's OWN predictions, so the ten members do NOT share
        # it -- on gaia-stars-mask-cont2 the per-coef moon lift spans
        # 0.991-1.012 across seeds and the mesospheric scalar 0.9997-1.0074.
        # Saving only member 0's and applying it to all ten made the RESTORED
        # ensemble a different predictor from the one that was validated:
        # moon amplitude MAD 0.01163 -> 0.01192, mesospheric ML 45.0 -> 45.56,
        # blue chi2 1.202 -> 1.238, while a fresh retrain reproduced the
        # original numbers EXACTLY (training is deterministic given the seeds).
        # The shipped artifact has to be the model that was measured.
        "member_jensen_corrections": [m["jensen_corrections"] for m in members],
        "coef_upper_bound":   ensemble_artifacts["coef_upper_bound"],
        "geom_kwargs":  ensemble_artifacts["geom_kwargs"],
        "group_indices":    ensemble_artifacts["group_indices"],
        "score_slices":     ensemble_artifacts["score_slices"],
        "group_score_dims": ensemble_artifacts["group_score_dims"],
        "n_input_score":    int(ensemble_artifacts["n_input_score"]),
        "coef_names": list(ensemble_artifacts["coef_names"]),
        "ctx_names":  list(ensemble_artifacts["ctx_names"]),
        # Constraint-derived amplitude rules (trainer 3.9).  These are FITTED
        # objects -- R, S, the basis integrals and the column indices -- not
        # config knobs, and `predict_sci_coefficients_default` silently skips a
        # rule whose object is absent.  Omitting them made a loaded ensemble
        # predict a moon amplitude 5x off with a +0.30 dex bias while the
        # in-session model was correct, and nothing raised.  `config` records
        # only whether each rule was ENABLED, which is not enough to apply it.
        "moon_down_amp_rule": first.get("moon_down_amp_rule"),
        "zodi_ceiling_rule":  first.get("zodi_ceiling_rule"),
        # Leinert zodi correction the training targets were anchored with.
        # Inference must compute zodi_po / moon_frac_po under the same one, or
        # the zodi-ceiling rule is applied against the wrong scale.  Optional:
        # files written before 2026-09-24 were all uncorrected ("none").
        "zodi_correction": str(ensemble_artifacts.get("zodi_correction", "none")),
        # Context-geometry version the model was trained on; see
        # mlp_predictor.data.CTX_GEOMETRY_VERSION.  Files without it are v1.
        "ctx_geometry_version": int(ensemble_artifacts.get("ctx_geometry_version", 1)),
        # Per-member training history (2026-09-23), for `diag.training_history()`
        # and any later post-hoc comparison of runs.
        #
        # This does NOT bump FORMAT_VERSION.  The version gate exists to refuse
        # files missing state that would silently MIS-PREDICT (that is what v1
        # -> v2 was about); history cannot change a prediction, so a file
        # without it is still a correct predictor and must stay loadable --
        # bumping would strand every ensemble trained before today.  The loader
        # therefore treats these keys as optional.
        #
        # Size: n_epochs x (2 losses + one alpha per group) x n_members, i.e.
        # ~300 x 8 x 10 floats ~ 200 kB against a 57 MB archive.
        "member_history": [list(m.get("history") or []) for m in members],
        "member_blend_history": [list(m.get("blend_history") or [])
                                 for m in members],
        # All-or-nothing: the loader indexes these by member position, so a
        # partially populated list would silently attach one member's best
        # epoch to another.  Empty is unambiguous; a short list is not.
        "best_epochs": ([int(m["best_epoch"]) for m in members]
                        if all(m.get("best_epoch") is not None for m in members)
                        else []),
        "best_val_losses": ([float(m["best_val_loss"]) for m in members]
                            if all(m.get("best_val_loss") is not None
                                   for m in members) else []),
    }
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    print(f"[save_ensemble] wrote {len(members)}-member ensemble to {out_path} "
          f"(format v{FORMAT_VERSION}).")
    return out_path


def _pick_device(explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model_from_config(cfg: Mapping[str, Any], *, n_input_score: int,
                             n_ctx: int, group_score_dims: Mapping[str, int],
                             ctx_names: list[str]) -> DualEncoderGroupHeadMLPCompressed:
    """Reconstruct a model instance with the architecture recorded at training time."""
    return DualEncoderGroupHeadMLPCompressed(
        n_score=int(n_input_score), n_ctx=int(n_ctx),
        group_score_dims=dict(group_score_dims),
        ctx_names=list(ctx_names),
        encoder_dims=tuple(int(v) for v in cfg["encoder_dims"]),
        ctx_dims=tuple(int(v) for v in cfg["ctx_dims"]),
        trunk_dims=tuple(int(v) for v in cfg["trunk_dims"]),
        head_dim=int(cfg["head_dim"]),
        zodi_head_extra_dims=tuple(int(v) for v in cfg["zodi_head_extra_dims"]),
        continuum_head_extra_dims=tuple(int(v) for v in cfg["continuum_head_extra_dims"]),
        continuum_branch_dims=tuple(int(v) for v in cfg["continuum_branch_dims"]),
        moon_zodi_coupling_dims=tuple(int(v) for v in cfg["moon_zodi_coupling_dims"]),
        # Scalar or per-group mapping; a restored ensemble must rebuild the
        # SAME model or its alphas start somewhere the training never saw.
        blend_init_alpha=(dict(cfg["blend_init_alpha"])
                          if isinstance(cfg["blend_init_alpha"], Mapping)
                          else float(cfg["blend_init_alpha"])),
        alpha_ctx_features=cfg["alpha_ctx_features"],
        zodi_ctx_restriction=cfg["zodi_ctx_restriction"],
        continuum_ctx_restriction=cfg["continuum_ctx_restriction"],
        moon_zodi_ctx_restriction=cfg["moon_zodi_ctx_restriction"],
    )


def load_ensemble(path: str | Path, *, device: str | None = None,
                  weights_only: bool = False) -> dict:
    """Load an ensemble saved with :func:`save_ensemble`.

    Parameters
    ----------
    path:
        Path to the ``.pt`` archive.
    device:
        Torch device string ('cpu', 'cuda', 'mps').  Defaults to the best
        available device.
    weights_only:
        Passed through to ``torch.load``.  Default False because the payload
        contains custom Python objects (``RobustScaler`` instances, dicts of
        numpy arrays) that ``torch.load(weights_only=True)`` would reject.

    Returns
    -------
    dict
        An ``mlp_artifacts`` dict shaped exactly like the one that
        :class:`~mlp_predictor.trainer.Trainer` returns.  Ready to hand to
        :func:`~mlp_predictor.trainer.predict_sci_coefficients_default`
        or :func:`~mlp_predictor.inference.predict_sky_from_minimal_inputs`.
    """
    dev = _pick_device(device)
    payload = torch.load(str(Path(path).expanduser()), map_location=dev,
                          weights_only=bool(weights_only))
    fv = int(payload.get("format_version", -1))
    if fv != FORMAT_VERSION:
        _extra = ""
        if fv == FORMAT_VERSION - 1:
            _extra = (
                "  v%d did not persist the constraint-derived amplitude rules "
                "(`moon_down_amp_rule`, `zodi_ceiling_rule`), so loading it "
                "would silently drop them and mis-predict the moon amplitude "
                "by ~5x on dark rows.  Re-save from a live training session "
                "(no retrain needed if the kernel still holds the artifacts) "
                "or retrain." % (FORMAT_VERSION - 1))
        raise ValueError(
            f"Unsupported format_version {fv!r}; expected {FORMAT_VERSION}.{_extra}")

    cfg = dict(payload["config"])
    ctx_names = list(payload["ctx_names"])
    from .data import CTX_GEOMETRY_VERSION as _CGV
    _geo_v = int(payload.get("ctx_geometry_version", 1))
    if _geo_v != _CGV:
        print(f"[load_ensemble] WARNING: this ensemble was trained on context "
              f"geometry v{_geo_v}, but the code now computes v{_CGV}. "
              + ("v1 had sun_sep / moon_sep as ICRS separations (the Sun and Moon "
                 "placed at the solar-system barycentre: moon_sep was really "
                 "180 - solar elongation) and a barycentric Sun longitude behind "
                 "zodi_log10_v. " if _geo_v == 1 else "")
              + "Its predictions will differ from the ones it was validated "
              "with, because it is now fed features it never saw. Retrain.")
    group_score_dims = dict(payload["group_score_dims"])
    n_input_score = int(payload["n_input_score"])

    members: list[dict] = []
    # Per-member calibration, with a fallback for files written before
    # 2026-09-12 that carry only member 0's.  Loading such a file gives a
    # predictor that differs measurably from the one that was validated
    # (moon amplitude MAD 2.5%, blue chi2 3%), so say so rather than silently
    # degrading -- retrain to regain exactness.
    _n_members = len(payload["member_state_dicts"])
    _member_jc = payload.get("member_jensen_corrections")
    if _member_jc is None or len(_member_jc) != _n_members:
        if _member_jc is not None:
            print(f"[load_ensemble] WARNING: member_jensen_corrections has "
                  f"{len(_member_jc)} entries for {_n_members} members; "
                  f"falling back to the shared copy.")
        else:
            print("[load_ensemble] NOTE: this file predates per-member "
                  "calibration (2026-09-12) and carries only member 0's "
                  "`jensen_corrections`; every member will use it. Predictions "
                  "will differ slightly from the trained-in-session model "
                  "(measured: moon amplitude MAD 2.5%, blue chi2 3%). Retrain "
                  "to regain exactness.")
        _member_jc = [payload["jensen_corrections"]] * _n_members
    # Training history is optional: files written before 2026-09-23 have none,
    # and their predictions are unaffected by that.  Say so once rather than
    # letting `diag.training_history()` fail with a KeyError further downstream.
    _hist = payload.get("member_history")
    _bhist = payload.get("member_blend_history")
    if not _hist:
        print("[load_ensemble] NOTE: this file predates persisted training "
              "history (2026-09-23); per-epoch loss curves are unavailable "
              "(predictions are unaffected). Retrain to record them.")
    _hist = list(_hist) if _hist else [[]] * _n_members
    _bhist = list(_bhist) if _bhist else [[]] * _n_members
    _best_ep = list(payload.get("best_epochs") or [])
    _best_vl = list(payload.get("best_val_losses") or [])
    for _i_member, sd in enumerate(payload["member_state_dicts"]):
        model = _build_model_from_config(
            cfg,
            n_input_score=n_input_score,
            n_ctx=len(ctx_names),
            group_score_dims=group_score_dims,
            ctx_names=ctx_names,
        ).to(dev)
        model.load_state_dict(sd)
        model.eval()
        members.append({
            "model": model,
            "device": dev,
            "score_scaler": payload["score_scaler"],
            "ctx_scaler":   payload["ctx_scaler"],
            "compressors":  payload["compressors"],
            "jensen_corrections": _member_jc[_i_member],
            "moon_down_amp_rule": payload.get("moon_down_amp_rule"),
            "zodi_ceiling_rule":  payload.get("zodi_ceiling_rule"),
            "zodi_correction": str(payload.get("zodi_correction", "none")),
            "coef_upper_bound":   payload["coef_upper_bound"],
            "geom_kwargs":  payload["geom_kwargs"],
            "group_indices":    payload["group_indices"],
            "score_slices":     payload["score_slices"],
            "group_score_dims": group_score_dims,
            "n_input_score":    n_input_score,
            "coef_names": list(payload["coef_names"]),
            "ctx_names":  ctx_names,
            "config": cfg,
            "history": _hist[_i_member] if _i_member < len(_hist) else [],
            "blend_history": _bhist[_i_member] if _i_member < len(_bhist) else [],
            "best_epoch": (_best_ep[_i_member]
                           if _i_member < len(_best_ep) else None),
            "best_val_loss": (_best_vl[_i_member]
                              if _i_member < len(_best_vl) else None),
        })

    print(f"[load_ensemble] restored {len(members)}-member ensemble from {path} "
          f"onto device={dev}.")
    return {
        "is_ensemble": True,
        "seeds": list(payload["seeds"]),
        "members": members,
        "compressors":  payload["compressors"],
        "coef_upper_bound":   payload["coef_upper_bound"],
        "geom_kwargs":  payload["geom_kwargs"],
        "group_indices":    payload["group_indices"],
        "score_slices":     payload["score_slices"],
        "group_score_dims": group_score_dims,
        "n_input_score":    n_input_score,
        "coef_names": list(payload["coef_names"]),
        "ctx_names":  ctx_names,
        "config": cfg,
        # Same keys the trainer puts on a live artifact, so the history
        # diagnostic does not care whether the ensemble was trained in this
        # session or restored from disk.
        "best_epochs": _best_ep,
        "best_val_losses": _best_vl,
        "zodi_correction": str(payload.get("zodi_correction", "none")),
        "ctx_geometry_version": _geo_v,
    }


__all__ = ["save_ensemble", "load_ensemble", "FORMAT_VERSION"]
