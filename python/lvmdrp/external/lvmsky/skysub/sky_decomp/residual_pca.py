"""PALACE-Aijc sky decomposition with a frozen full-grid residual PCA basis."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from .fit import LSF_CHANNELS, SPLIT_ZODI_CONTINUUM_DEFAULTS
from .lsf_spline2d import (
    LSF_OFFSET_BASIS_COUNT,
    _component_masses,
    native_pixel_edges,
)
from .moon_zodi_model import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PALACE_OH_SUFFIX,
    SKYFAR_LINEAR_RIDGE_LAMBDA,
    SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX,
    file_sha256,
    validate_decomposition_asset_contract,
    wave_sha256,
)
from .lsf_surface_iterative import continuum_fit_weights
from .telluric_corrected_lines import SkyDecompTelluricCorrectedLinesLSFSpline2D


RESIDUAL_PCA_ASSET = "residual_pca/palace_aijc_full_native_pca_v1.npz"
RESIDUAL_PCA_FIT_MODEL = "telluric-corrected-lines-palace-aijc-residual-pca"
LINE_AMPLITUDE_PCA_ASSET = (
    "residual_pca/palace_aijc_line_amplitude_pca_v1.npz"
)
LINE_AMPLITUDE_PCA_FIT_MODEL = (
    "telluric-corrected-lines-palace-aijc-line-amplitude-pca"
)
PALACE_AIJC_VN_FIT_MODEL = "telluric-corrected-lines-palace-aijc-vn"
VN_LINE_AMPLITUDE_PCA_ASSET = (
    "residual_pca/palace_aijc_vn_line_amplitude_pca_v1.npz"
)
VN_LINE_AMPLITUDE_PCA_FIT_MODEL = (
    "telluric-corrected-lines-palace-aijc-vn-line-amplitude-pca"
)
VNF_COEFFICIENT_PCA_PREP_ASSET = (
    "residual_pca/palace_aijc_vnf_coefficient_pca_prep_v1.npz"
)
VNF_COEFFICIENT_PCA_PREP_FIT_MODEL = (
    "telluric-corrected-lines-palace-aijc-vnf-coefficient-pca-prep"
)
VNF_COEFFICIENT_LINE_AMPLITUDE_PCA_ASSET = (
    "residual_pca/palace_aijc_vnf_coefficient_line_amplitude_pca_v1.npz"
)
VNF_COEFFICIENT_LINE_AMPLITUDE_PCA_FIT_MODEL = (
    "telluric-corrected-lines-palace-aijc-vnf-coefficient-line-amplitude-pca"
)
VNF_LINE_ADJOINT_PCA_ASSET = (
    "residual_pca/palace_aijc_vnf_line_adjoint_pca_v1.npz"
)
VNF_LINE_ADJOINT_PCA_FIT_MODEL = (
    "telluric-corrected-lines-palace-aijc-vnf-line-adjoint-pca"
)
SPLIT_ZODI_VNF_LINE_AMPLITUDE_PCA_ASSET = (
    "residual_pca/palace_aijc_vnf_split_zodi_line_amplitude_pca30_v1.npz"
)


def _validate_method_asset(root: Path, contract_key: str, label: str) -> None:
    if (root / "bundle_manifest.json").is_file():
        validate_decomposition_asset_contract(str(root), contract_key, label)


@lru_cache(maxsize=None)
def _load_basis(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        wave = np.asarray(data["wave"], dtype=np.float64)
        mean = np.asarray(data["residual_mean"], dtype=np.float64)
        components = np.asarray(data["components"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))
    if (
        wave.ndim != 1
        or mean.shape != wave.shape
        or components.shape != (30, wave.size)
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(components))
        or np.any(np.diff(wave) <= 0.0)
    ):
        raise ValueError("The residual PCA asset has an invalid native-grid shape")
    if metadata.get("wave_sha256") != wave_sha256(wave):
        raise ValueError("The residual PCA asset wavelength hash is invalid")
    source_oh = Path(path).parent.parent / str(metadata.get("source_oh_asset", ""))
    if (
        not source_oh.is_file()
        or metadata.get("source_oh_asset_sha256") != file_sha256(source_oh)
    ):
        raise ValueError("The residual PCA basis and PALACE OH source do not match")
    if not np.allclose(components @ components.T, np.eye(30), atol=2.0e-12):
        raise ValueError("The residual PCA components are not orthonormal")
    wave.setflags(write=False)
    mean.setflags(write=False)
    components.setflags(write=False)
    return wave, mean, components, metadata


@lru_cache(maxsize=None)
def _load_line_amplitude_basis(
    path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    with np.load(path, allow_pickle=False) as data:
        wave = np.asarray(data["wave"], dtype=np.float64)
        line_names = np.asarray(data["line_names"])
        line_wave = np.asarray(data["line_wave"], dtype=np.float64)
        line_group = np.asarray(data["line_group"], dtype=np.int64)
        mean = np.asarray(data["amplitude_mean"], dtype=np.float64)
        components = np.asarray(data["components"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))
    if (
        wave.ndim != 1
        or line_names.ndim != 1
        or line_wave.shape != line_names.shape
        or line_group.shape != line_names.shape
        or mean.shape != line_names.shape
        or components.ndim != 2
        or components.shape[0] < 1
        or components.shape[1] != line_names.size
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(components))
        or np.any(np.diff(wave) <= 0.0)
    ):
        raise ValueError("The line-amplitude PCA asset has an invalid shape")
    if metadata.get("wave_sha256") != wave_sha256(wave):
        raise ValueError("The line-amplitude PCA asset wavelength hash is invalid")
    names_hash = hashlib.sha256("\n".join(line_names.tolist()).encode()).hexdigest()
    if metadata.get("line_names_sha256") != names_hash:
        raise ValueError("The line-amplitude PCA line-name hash is invalid")
    source_assets = metadata.get("source_assets_sha256")
    if not isinstance(source_assets, dict) or not source_assets:
        raise ValueError("The line-amplitude PCA source-asset contract is missing")
    root = Path(path).parent.parent
    for relative, expected in source_assets.items():
        source = root / str(relative)
        if not source.is_file() or file_sha256(source) != str(expected):
            raise ValueError(
                f"The line-amplitude PCA source asset does not match: {relative}"
            )
    if not np.allclose(
        components @ components.T,
        np.eye(components.shape[0]),
        atol=3.0e-12,
    ):
        raise ValueError("The line-amplitude PCA components are not orthonormal")
    for array in (wave, line_names, line_wave, line_group, mean, components):
        array.setflags(write=False)
    return wave, line_names, line_wave, line_group, mean, components, metadata


@lru_cache(maxsize=None)
def _load_oh_coefficient_basis(
    path: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    with np.load(path, allow_pickle=False) as data:
        wave = np.asarray(data["wave"], dtype=np.float64)
        names = np.asarray(data["coefficient_names"])
        mean = np.asarray(data["coefficient_mean"], dtype=np.float64)
        components = np.asarray(data["components"], dtype=np.float64)
        metadata = json.loads(str(data["metadata_json"].item()))
    if (
        wave.ndim != 1
        or names.shape != mean.shape
        or mean.shape != (357,)
        or components.ndim != 2
        or components.shape[0] < 1
        or components.shape[1] != mean.size
        or np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(components))
        or np.any(np.diff(wave) <= 0.0)
    ):
        raise ValueError("The VNF OH-coefficient PCA asset has an invalid shape")
    if metadata.get("wave_sha256") != wave_sha256(wave):
        raise ValueError("The VNF OH-coefficient PCA wavelength hash is invalid")
    source_assets = metadata.get("source_assets_sha256")
    if not isinstance(source_assets, dict) or not source_assets:
        raise ValueError("The VNF OH-coefficient PCA source contract is missing")
    root = Path(path).parent.parent
    for relative, expected in source_assets.items():
        source = root / str(relative)
        if not source.is_file() or file_sha256(source) != str(expected):
            raise ValueError(
                f"The VNF OH-coefficient PCA source asset does not match: {relative}"
            )
    if not np.allclose(
        components @ components.T,
        np.eye(components.shape[0]),
        atol=3.0e-12,
    ):
        raise ValueError("The VNF OH-coefficient PCA components are not orthonormal")
    for array in (wave, names, mean, components):
        array.setflags(write=False)
    return wave, names, mean, components, metadata


def _individual_line_names(model) -> np.ndarray:
    names = np.empty(model._line_wave.size, dtype="U32")
    for family, group_slice in model._group_slices.items():
        use = (model._line_group >= group_slice.start) & (
            model._line_group < group_slice.stop
        )
        names[use] = [
            f"{family.upper()}_{index:05d}" for index in range(np.count_nonzero(use))
        ]
    if np.any(names == ""):
        raise ValueError("The individual line catalog contains an unknown family")
    return names


def _individual_line_design(model) -> sp.csc_matrix:
    """Return exact integrated profiles for every retained individual line."""
    if model.lsf_surface_state is None:
        raise ValueError("An LSF surface is required for individual line profiles")
    n_lines = model._line_wave.size
    transmission = model._line_transmission(model._line_wave)
    blocks = []
    for channel, _, _ in LSF_CHANNELS:
        indices = model._line_indices[channel]
        masses = _component_masses(
            model.lsf_surface_state,
            channel,
            model._line_wave[indices],
        )
        transform = sp.coo_matrix(
            (
                (masses * transmission[indices, None]).ravel(),
                (
                    np.arange(indices.size * LSF_OFFSET_BASIS_COUNT),
                    np.repeat(indices, LSF_OFFSET_BASIS_COUNT),
                ),
            ),
            shape=(indices.size * LSF_OFFSET_BASIS_COUNT, n_lines),
        ).tocsc()
        blocks.append(model._line_components[channel] @ transform)
    design = sum(blocks[1:], start=blocks[0]).tocsc()
    widths = np.diff(native_pixel_edges(model.wave))
    integral = np.asarray(widths @ design).ravel()
    active = (
        (model._line_wave >= model.wave[0])
        & (model._line_wave <= model.wave[-1])
        & (integral > 0.0)
    )
    scale = np.zeros(n_lines)
    scale[active] = 1.0 / integral[active]
    return (design @ sp.diags(scale)).tocsc()


class SkyDecompTelluricCorrectedLinesPalaceAijc(
    SkyDecompTelluricCorrectedLinesLSFSpline2D
):
    """Use ``Aijc`` OH ratios from the selected PALACE-format table."""

    def __init__(self, wave: np.ndarray, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("palace_oh_suffix") is None:
            kwargs["palace_oh_suffix"] = DEFAULT_PALACE_OH_SUFFIX
        super().__init__(wave, *args, **kwargs)

    @staticmethod
    def _oh_amplitude(group) -> np.ndarray:
        return np.asarray(group["Aijc"] * group["gi"], dtype=float)


class SkyDecompTelluricCorrectedLinesPalaceAijcVNFPCAPrep(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """Fit VNF OH amplitudes through their frozen coefficient-PCA subspace."""

    oh_coefficient_pca_asset = VNF_COEFFICIENT_PCA_PREP_ASSET
    qp_retry_static_regularization_constant = 1.0e-6

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_oh_pca_components: int | None = None,
        **kwargs: Any,
    ) -> None:
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(
            root,
            "vnf_coefficient_pca_prep_contract",
            "VNF coefficient PCA Prep",
        )
        basis_wave, names, mean, components, metadata = _load_oh_coefficient_basis(
            str(root / self.oh_coefficient_pca_asset)
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError(
                "VNF OH-coefficient PCA requires its exact full native wavelength grid"
            )
        selected = int(metadata.get("selected_components", components.shape[0]))
        count = selected if n_oh_pca_components is None else int(n_oh_pca_components)
        if count < 1 or count > components.shape[0]:
            raise ValueError(
                f"n_oh_pca_components must be between 1 and {components.shape[0]}"
            )
        self.n_oh_pca_components = count
        self.oh_coefficient_pca_metadata = metadata
        self._oh_group_names = names
        self._oh_coefficient_basis = np.vstack((mean, components[:count]))
        super().__init__(wave, *args, **kwargs)
        expected_names = np.asarray([f"OH_{index:03d}" for index in range(357)])
        if not np.array_equal(names, expected_names):
            raise ValueError("The VNF OH-coefficient PCA group order changed")
        self.design_names[: self._oh_coefficient_basis.shape[0]] = [
            "OHVNFPCAPrep_mean",
            *[
                f"OHVNFPCAPrep_{index:03d}"
                for index in range(1, self.n_oh_pca_components + 1)
            ],
        ]

    def _build_oh(self) -> np.ndarray:
        grouped = super()._build_oh()
        if grouped.shape[0] != self._oh_coefficient_basis.shape[1]:
            raise ValueError("The VNF OH group count changed")
        self._oh_group_matrix_stick = self.matrix_oh_stick.copy()
        self.matrix_oh_stick = self._oh_coefficient_basis @ self.matrix_oh_stick
        return self._oh_coefficient_basis @ grouped

    def _rebuild_family(self, family: str) -> None:
        super()._rebuild_family(family)
        if family == "oh":
            self._oh_group_matrix_stick = self.matrix_oh_stick.copy()
            self.matrix_oh = self._oh_coefficient_basis @ self.matrix_oh
            self.matrix_oh_stick = self._oh_coefficient_basis @ self.matrix_oh_stick

    def _fit_design(self, design_matrix, flux, ivar, **kwargs):
        explicit = kwargs.pop("unconstrained_indices", None)
        n_rows = int(np.shape(design_matrix)[0])
        n_oh = int(self._oh_coefficient_basis.shape[0])
        n_lines = n_oh + self.matrix_atom.shape[0] + self.matrix_orc.shape[0] + 1
        free = np.asarray([] if explicit is None else explicit, dtype=int)
        if n_rows in (n_lines, len(self.design_names)) or (
            explicit is not None and n_rows > len(self.design_names)
        ):
            free = np.union1d(free, np.arange(1, n_oh, dtype=int))
        return super()._fit_design(
            design_matrix,
            flux,
            ivar,
            unconstrained_indices=free if free.size else None,
            **kwargs,
        )

    def _line_source(self, line_coefficient: np.ndarray) -> np.ndarray:
        line_coefficient = np.asarray(line_coefficient, dtype=float)
        n_oh = self._oh_coefficient_basis.shape[0]
        grouped_oh = self._oh_coefficient_basis.T @ line_coefficient[:n_oh]
        grouped = np.concatenate((grouped_oh, line_coefficient[n_oh:]))
        if grouped.shape != (self._n_groups,):
            raise ValueError(
                "Expanded VNF line coefficient has an invalid shape: "
                f"latent={line_coefficient.shape}, expanded={grouped.shape}, "
                f"catalog={(self._n_groups,)}"
            )
        self._active_line_coefficient = grouped
        sticks = np.vstack(
            (
                self._oh_group_matrix_stick,
                self.matrix_atom_stick,
                self.matrix_orc_stick,
                self.matrix_o2_stick,
            )
        )
        return sticks.T @ grouped

    def _assemble_refined_matrices(self) -> dict[str, np.ndarray]:
        matrices = super()._assemble_refined_matrices()
        matrices["oh"] = self._oh_coefficient_basis @ matrices["oh"]
        return matrices

    def _finalize_result(self, *args, **kwargs):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += (
            f" | oh_vnf_pca_prep={self.n_oh_pca_components}"
            " | oh_vnf_pca_mean_scale=True"
            " | oh_vnf_pca_scores_signed=True"
            " | qp_insufficient_progress_retry="
            f"{self.qp_retry_static_regularization_constant:g}"
            " | oh_strength=PALACE_Aijc"
        )
        self.fit_summary = result.fit_summary
        return result


class SkyDecompTelluricCorrectedLinesPalaceAijcVN(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """Tie OH amplitudes only by upper vibrational and rotational levels."""

    oh_group_keys = ("v_upper", "N_upper")

    def __init__(self, wave: np.ndarray, *args: Any, **kwargs: Any) -> None:
        super().__init__(wave, *args, **kwargs)
        count = self.matrix_oh.shape[0]
        self.design_names[:count] = [f"OHVN_{index:03d}" for index in range(count)]

    def _finalize_result(self, *args, **kwargs):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += (
            f" | oh_grouping=v_upper,N_upper | oh_groups={self.matrix_oh.shape[0]}"
            " | oh_strength=PALACE_Aijc"
        )
        self.fit_summary = result.fit_summary
        return result


class SkyDecompTelluricCorrectedLinesResidualPCA(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """Add a frozen 10-, 20-, or 30-component residual PCA basis to PALACE Aijc."""

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_residual_pca_components: int = 20,
        **kwargs: Any,
    ) -> None:
        count = int(n_residual_pca_components)
        if count not in (10, 20, 30):
            raise ValueError("n_residual_pca_components must be 10, 20, or 30")
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(root, "residual_pca_contract", "residual PCA")
        basis_wave, mean, components, metadata = _load_basis(
            str(root / RESIDUAL_PCA_ASSET)
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError("Residual PCA requires its exact full native wavelength grid")
        self.n_residual_pca_components = count
        self.residual_pca_metadata = metadata
        self._residual_mean = mean
        self._residual_components = components[:count]
        super().__init__(wave, *args, **kwargs)

    def _build_diffuse(self) -> tuple[np.ndarray, list[str]]:
        matrix, names = super()._build_diffuse()
        self._physical_diffuse_rows = matrix.shape[0]
        pca = np.vstack((self._residual_mean, self._residual_components))
        return np.vstack((matrix, pca)), names + [
            "ResidualPCA_mean",
            *[
                f"ResidualPCA_{index:02d}"
                for index in range(1, self.n_residual_pca_components + 1)
            ],
        ]

    def _fit_design(self, design_matrix, flux, ivar, **kwargs):
        explicit = kwargs.pop("unconstrained_indices", None)
        n_rows = int(np.shape(design_matrix)[0])
        n_diffuse = int(self.matrix_diffuse.shape[0])
        n_continuum = self.matrix_moon.shape[0] + self.matrix_zodi.shape[0] + n_diffuse
        n_full = len(self.design_names)
        unconstrained = explicit
        if unconstrained is None and n_rows == n_continuum:
            stop = n_continuum
            unconstrained = np.arange(
                stop - self.n_residual_pca_components, stop, dtype=int
            )
        elif unconstrained is None and n_rows == n_full:
            diffuse = self._component_slices(
                self._matrix_bundle(
                    self.matrix_oh,
                    self.matrix_moon,
                    self.matrix_diffuse,
                    self.matrix_atom,
                    self.matrix_orc,
                    self.matrix_o2,
                    matrix_zodi=self.matrix_zodi,
                )
            )["diffuse"]
            unconstrained = np.arange(
                diffuse.stop - self.n_residual_pca_components,
                diffuse.stop,
                dtype=int,
            )
        return super()._fit_design(
            design_matrix,
            flux,
            ivar,
            unconstrained_indices=unconstrained,
            **kwargs,
        )

    def _fit_continuum_stage(self, run, flux, ivar, skyline_mask):
        weights, channel_noise = continuum_fit_weights(
            self.wave,
            flux - run.continuum - run.line_model,
            ivar,
            skyline_mask,
            line_weight=self.config.line_weight,
            huber_transition_sigma=self.config.huber_transition_sigma,
        )
        physical_diffuse = run.matrices["diffuse"][: self._physical_diffuse_rows]
        parts = [run.matrices["moon"]]
        if self.split_zodi:
            parts.append(run.matrices["zodi"])
        parts.append(physical_diffuse)
        physical_design = np.vstack(parts)
        n_moon = run.matrices["moon"].shape[0]
        n_zodi = run.matrices["zodi"].shape[0] if self.split_zodi else 0
        physical_fit = super()._fit_design(
            physical_design,
            flux - run.line_model,
            weights,
            moon_slice=slice(0, n_moon),
            zodi_slice=(slice(n_moon, n_moon + n_zodi) if n_zodi else None),
            diffuse_slice=slice(n_moon + n_zodi, physical_design.shape[0]),
        )
        if str(physical_fit["status"]) not in self._SOLVED:
            return physical_fit, None, None, None, weights, channel_noise

        physical_coefficient = np.asarray(physical_fit["coef"], dtype=float)
        physical_coef_err = np.asarray(physical_fit["coef_err"], dtype=float)
        physical_continuum = physical_design.T @ physical_coefficient
        pca_design = run.matrices["diffuse"][self._physical_diffuse_rows :]
        pca_fit = super()._fit_design(
            pca_design,
            flux - run.line_model - physical_continuum,
            ivar,
            unconstrained_indices=np.arange(1, pca_design.shape[0], dtype=int),
        )
        if str(pca_fit["status"]) not in self._SOLVED:
            return pca_fit, None, None, None, weights, channel_noise

        coefficient = np.concatenate((physical_coefficient, pca_fit["coef"]))
        coef_err = np.concatenate((physical_coef_err, pca_fit["coef_err"]))
        continuum = physical_continuum + pca_design.T @ pca_fit["coef"]
        pca_fit["coef_cov_moon"] = physical_fit.get("coef_cov_moon")
        pca_fit["coef_cov_zodi"] = physical_fit.get("coef_cov_zodi")
        return pca_fit, coefficient, coef_err, continuum, weights, channel_noise

    def _components_from_coef(self, coef, matrices):
        components = super()._components_from_coef(coef, matrices)
        diffuse = self._component_slices(matrices)["diffuse"]
        pca_start = diffuse.start + self._physical_diffuse_rows
        components["residual_pca"] = (
            matrices["diffuse"][self._physical_diffuse_rows :].T
            @ np.asarray(coef)[pca_start : diffuse.stop]
        )
        components["diffuse"] = components["ho2"] + components["feo"] + components["o2ac"]
        return components

    def _continuum_from_components(self, components):
        return super()._continuum_from_components(components) + components["residual_pca"]

    def _finalize_result(self, *args, **kwargs):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += (
            f" | residual_pca={self.n_residual_pca_components}"
            " | oh_strength=PALACE_Aijc"
        )
        self.fit_summary = result.fit_summary
        return result


class SkyDecompTelluricCorrectedLinesLineAmplitudePCA(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """Add a signed PCA correction in individual-line amplitude space."""

    line_amplitude_pca_asset = LINE_AMPLITUDE_PCA_ASSET
    line_amplitude_pca_contract = "line_amplitude_pca_contract"

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_line_amplitude_pca_components: int = 20,
        **kwargs: Any,
    ) -> None:
        count = int(n_line_amplitude_pca_components)
        if count not in (10, 20, 30):
            raise ValueError(
                "n_line_amplitude_pca_components must be 10, 20, or 30"
            )
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(
            root,
            self.line_amplitude_pca_contract,
            "line-amplitude PCA",
        )
        basis_wave, line_names, line_wave, line_group, mean, components, metadata = (
            _load_line_amplitude_basis(str(root / self.line_amplitude_pca_asset))
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError(
                "Line-amplitude PCA requires its exact full native wavelength grid"
            )
        self.n_line_amplitude_pca_components = count
        self.line_amplitude_pca_metadata = metadata
        self._line_amplitude_names = line_names
        self._line_amplitude_wave = line_wave
        self._line_amplitude_group = line_group
        self._line_amplitude_mean = mean
        self._line_amplitude_components = components[:count]
        super().__init__(wave, *args, **kwargs)
        if not np.array_equal(_individual_line_names(self), line_names):
            raise ValueError("The line-amplitude PCA line list does not match the model")
        if not np.array_equal(self._line_wave, line_wave) or not np.array_equal(
            self._line_group, line_group
        ):
            raise ValueError("The individual line-amplitude catalog order changed")

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ):
        started = time.perf_counter()
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        result = super().fit(flux, ivar, verbose=verbose)
        if result.fit_status not in self._SOLVED:
            return result

        line_design = _individual_line_design(self)
        amplitude_basis = np.vstack(
            (self._line_amplitude_mean, self._line_amplitude_components)
        )
        correction_design = np.asarray(line_design @ amplitude_basis.T).T
        correction_fit = self._fit_design(
            correction_design,
            flux - result.bestfit_lsf,
            ivar,
            unconstrained_indices=np.arange(correction_design.shape[0]),
        )
        correction_status = str(correction_fit["status"])
        if correction_status not in self._SOLVED:
            result.fit_status = f"failed:line_amplitude_pca:{correction_status}"
            result.fit_summary += f" | line_amplitude_pca={correction_status}"
            return result

        coefficient = np.asarray(correction_fit["coef"], dtype=float)
        coefficient_err = np.asarray(correction_fit["coef_err"], dtype=float)
        correction = correction_design.T @ coefficient
        residual = flux - result.bestfit_lsf - correction
        names = [
            "LineAmplitudePCA_mean",
            *[
                f"LineAmplitudePCA_{index:02d}"
                for index in range(1, self.n_line_amplitude_pca_components + 1)
            ],
        ]

        result.coef = np.concatenate((result.coef, coefficient))
        result.coef_err = np.concatenate((result.coef_err, coefficient_err))
        result.design_names = [*result.design_names, *names]
        result.components = dict(result.components)
        result.components["line_amplitude_pca"] = correction
        result.bestfit_lsf = result.bestfit_lsf + correction
        result.resid = residual
        result.resid_level = -3.0 * float(np.nanstd(residual))
        if np.all(np.isfinite(coefficient_err)):
            correction_sigma = np.sqrt(
                np.sum((correction_design.T * coefficient_err) ** 2, axis=1)
            )
            result.bestfit_lsf_sigma = np.sqrt(
                result.bestfit_lsf_sigma**2 + correction_sigma**2
            )
        else:
            result.bestfit_lsf_sigma = np.full_like(residual, np.nan)

        valid = np.isfinite(residual) & np.isfinite(ivar) & (ivar > 0.0)
        chi2 = float(np.sum(residual[valid] ** 2 * ivar[valid]))
        result.reduced_chi2 = chi2 / max(int(np.sum(valid)) - result.coef.size, 1)
        result.rms_resid = float(np.sqrt(np.mean(residual[valid] ** 2)))
        mean_flux = float(np.average(flux[valid], weights=ivar[valid]))
        total_sum = float(np.sum((flux[valid] - mean_flux) ** 2 * ivar[valid]))
        result.r2 = 1.0 - chi2 / total_sum if total_sum > 0.0 else np.nan
        result.fit_elapsed_sec = time.perf_counter() - started
        result.fit_summary += (
            f" | line_amplitude_pca={self.n_line_amplitude_pca_components}"
            f" | line_amplitude_pca_status={correction_status}"
            " | line_amplitude_pca_signed=True"
            f" | individual_lines={line_design.shape[1]}"
            " | line_amplitude_pca_lsf=fixed_final"
        )

        self.coef = result.coef
        self.coef_err = result.coef_err
        self.bestfit_lsf = result.bestfit_lsf
        self.final_line_model = self.final_line_model + correction
        self.rms_resid = result.rms_resid
        self.r2 = result.r2
        self.fit_summary = result.fit_summary
        return result


def _joint_line_amplitude_pca_fit(
    model,
    result,
    flux: np.ndarray,
    ivar: np.ndarray,
    started: float,
    summary_key: str,
    name_width: int,
    *,
    design_prefix: str = "LineAmplitudePCA",
    component_key: str = "line_amplitude_pca",
):
    if result.fit_status not in model._SOLVED:
        return result
    line_design = _individual_line_design(model)
    amplitude_basis = np.vstack(
        (model._line_amplitude_mean, model._line_amplitude_components)
    )
    correction_design = np.asarray(line_design @ amplitude_basis.T).T
    matrices = model._assemble_refined_matrices()
    keys = (
        ("oh", "moon", "zodi", "diffuse", "atom", "orc", "o2")
        if model.split_zodi
        else ("oh", "moon", "diffuse", "atom", "orc", "o2")
    )
    physical_design = model._stack_matrices(matrices, keys)
    joint_design = np.vstack((physical_design, correction_design))
    slices = model._component_slices(matrices)
    physical_count = physical_design.shape[0]
    joint_fit = model._fit_design(
        joint_design,
        flux,
        ivar,
        moon_slice=slices["moon"],
        zodi_slice=slices.get("zodi"),
        diffuse_slice=slices["diffuse"],
        unconstrained_indices=np.arange(
            physical_count,
            joint_design.shape[0],
            dtype=int,
        ),
    )
    joint_status = str(joint_fit["status"])
    if joint_status not in model._SOLVED:
        result.fit_status = f"failed:{summary_key}:{joint_status}"
        result.fit_summary += f" | {summary_key}={joint_status}"
        return result

    coefficient = np.asarray(joint_fit["coef"], dtype=float)
    coefficient_err = np.asarray(joint_fit["coef_err"], dtype=float)
    physical_coefficient = coefficient[:physical_count]
    pca_coefficient = coefficient[physical_count:]
    components = model._components_from_coef(physical_coefficient, matrices)
    correction = correction_design.T @ pca_coefficient
    components[component_key] = correction
    continuum = model._continuum_from_components(components)
    line_model = (
        components["oh"]
        + components["atom"]
        + components["orc"]
        + components["o2"]
        + correction
    )
    bestfit = continuum + line_model
    residual = flux - bestfit
    names = [
        f"{design_prefix}_mean",
        *[
            f"{design_prefix}_{index:0{name_width}d}"
            for index in range(1, model.n_line_amplitude_pca_components + 1)
        ],
    ]

    result.coef = coefficient
    result.coef_err = coefficient_err
    result.design_names = [*model.design_names, *names]
    result.components = components
    result.bestfit_lsf = bestfit
    result.resid = residual
    result.resid_level = -3.0 * float(np.nanstd(residual))
    if np.all(np.isfinite(coefficient_err)):
        result.bestfit_lsf_sigma = np.sqrt(
            np.sum((joint_design.T * coefficient_err) ** 2, axis=1)
        )
    else:
        result.bestfit_lsf_sigma = np.full_like(residual, np.nan)

    valid = np.isfinite(residual) & np.isfinite(ivar) & (ivar > 0.0)
    chi2 = float(np.sum(residual[valid] ** 2 * ivar[valid]))
    result.reduced_chi2 = chi2 / max(int(np.sum(valid)) - coefficient.size, 1)
    result.rms_resid = float(np.sqrt(np.mean(residual[valid] ** 2)))
    mean_flux = float(np.average(flux[valid], weights=ivar[valid]))
    total_sum = float(np.sum((flux[valid] - mean_flux) ** 2 * ivar[valid]))
    result.r2 = 1.0 - chi2 / total_sum if total_sum > 0.0 else np.nan
    result.fit_elapsed_sec = time.perf_counter() - started
    result.coef_cov_moon = joint_fit.get("coef_cov_moon")
    result.coef_cov_zodi = joint_fit.get("coef_cov_zodi")
    result.fit_summary += (
        f" | {summary_key}={model.n_line_amplitude_pca_components}"
        f" | {summary_key}_status={joint_status}"
        f" | {summary_key}_signed=True"
        f" | {summary_key}_joint_final=True"
        f" | individual_lines={line_design.shape[1]}"
        f" | {summary_key}_lsf=fixed_final"
    )

    model.coef = result.coef
    model.coef_err = result.coef_err
    model.bestfit_lsf = result.bestfit_lsf
    model.final_continuum = continuum
    model.final_line_model = line_model
    model.rms_resid = result.rms_resid
    model.r2 = result.r2
    model.fit_summary = result.fit_summary
    return result


class SkyDecompPalaceAijcVNFLineAmplitudePCA(
    SkyDecompTelluricCorrectedLinesLineAmplitudePCA
):
    """Jointly fit PALACE VNF OH and LSF-projected line-amplitude PCA."""

    oh_group_keys = ("v_upper", "N_upper", "F_upper")

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_line_amplitude_pca_components: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            wave,
            *args,
            n_line_amplitude_pca_components=n_line_amplitude_pca_components,
            **kwargs,
        )

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ):
        started = time.perf_counter()
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        result = SkyDecompTelluricCorrectedLinesPalaceAijc.fit(
            self,
            flux,
            ivar,
            verbose=verbose,
        )
        return _joint_line_amplitude_pca_fit(
            self,
            result,
            flux,
            ivar,
            started,
            "palace_vnf_line_amplitude_pca",
            2,
        )


class SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """PALACE VNF lines with the production split-zodi continuum."""

    oh_group_keys = ("v_upper", "N_upper", "F_upper")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **(SPLIT_ZODI_CONTINUUM_DEFAULTS | kwargs))

    def _finalize_result(self, *args: Any, **kwargs: Any):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += (
            " | oh_grouping=v_upper,N_upper,F_upper"
            " | continuum_profile=split-zodi-production-v1"
        )
        self.fit_summary = result.fit_summary
        return result


class SkyDecompPalaceCorrAijcVNFSplitZodiLSFSpline2D(
    SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D
):
    """Use the validation-selected SkyFar linear-ridge OH coefficients."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        requested = kwargs.get("palace_oh_suffix")
        if requested not in (None, SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX):
            raise ValueError(
                "palacecorr requires pmd_popmodel_OH"
                f"{SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX}.dat"
            )
        kwargs["palace_oh_suffix"] = SKYFAR_LINEAR_RIDGE_PALACE_OH_SUFFIX
        super().__init__(*args, **kwargs)

    def _finalize_result(self, *args: Any, **kwargs: Any):
        result = super()._finalize_result(*args, **kwargs)
        result.fit_summary += (
            " | oh_coefficients=skyfar-linear"
            f" | oh_coefficient_ridge_lambda={SKYFAR_LINEAR_RIDGE_LAMBDA:g}"
        )
        self.fit_summary = result.fit_summary
        return result


class SkyDecompPalaceAijcVNFSplitZodiLineAmplitudePCA30(
    SkyDecompPalaceAijcVNFLineAmplitudePCA
):
    """PALACE VNF plus PCA30 with production split-zodi continuum and 2-D LSF."""

    line_amplitude_pca_asset = SPLIT_ZODI_VNF_LINE_AMPLITUDE_PCA_ASSET
    line_amplitude_pca_contract = "split_zodi_vnf_line_amplitude_pca_contract"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs = SPLIT_ZODI_CONTINUUM_DEFAULTS | kwargs
        count = int(kwargs.get("n_line_amplitude_pca_components", 30))
        if count != 30:
            raise ValueError("The integrated production method requires PCA30")
        super().__init__(*args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any):
        result = super().fit(*args, **kwargs)
        result.fit_summary += " | continuum_profile=split-zodi-production-v1"
        self.fit_summary = result.fit_summary
        return result


# Compatibility for executed notebooks and frozen training provenance.
NIV_VNF_LINE_AMPLITUDE_PCA_ASSET = SPLIT_ZODI_VNF_LINE_AMPLITUDE_PCA_ASSET
SkyDecompPalaceAijcVNFNivContinuumLSFSpline2D = (
    SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D
)
SkyDecompPalaceAijcVNFNivContinuumLineAmplitudePCA = (
    SkyDecompPalaceAijcVNFSplitZodiLineAmplitudePCA30
)


class SkyDecompTelluricCorrectedLinesVNLineAmplitudePCA(
    SkyDecompTelluricCorrectedLinesPalaceAijcVN
):
    """Jointly fit the VN baseline and signed individual-line amplitude PCA."""

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_line_amplitude_pca_components: int = 20,
        **kwargs: Any,
    ) -> None:
        count = int(n_line_amplitude_pca_components)
        if count not in (10, 20, 30):
            raise ValueError(
                "n_line_amplitude_pca_components must be 10, 20, or 30"
            )
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(
            root,
            "vn_line_amplitude_pca_contract",
            "VN line-amplitude PCA",
        )
        basis_wave, line_names, line_wave, line_group, mean, components, metadata = (
            _load_line_amplitude_basis(str(root / VN_LINE_AMPLITUDE_PCA_ASSET))
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError(
                "VN line-amplitude PCA requires its exact full native wavelength grid"
            )
        self.n_line_amplitude_pca_components = count
        self.line_amplitude_pca_metadata = metadata
        self._line_amplitude_names = line_names
        self._line_amplitude_wave = line_wave
        self._line_amplitude_group = line_group
        self._line_amplitude_mean = mean
        self._line_amplitude_components = components[:count]
        super().__init__(wave, *args, **kwargs)
        if not np.array_equal(_individual_line_names(self), line_names):
            raise ValueError("The VN line-amplitude PCA line list does not match the model")
        if not np.array_equal(self._line_wave, line_wave) or not np.array_equal(
            self._line_group, line_group
        ):
            raise ValueError("The VN individual-line catalog order changed")

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ):
        started = time.perf_counter()
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        result = super().fit(flux, ivar, verbose=verbose)
        return _joint_line_amplitude_pca_fit(
            self,
            result,
            flux,
            ivar,
            started,
            "vn_line_amplitude_pca",
            2,
        )


class SkyDecompTelluricCorrectedLinesVNFPCALineAmplitudePCA(
    SkyDecompTelluricCorrectedLinesPalaceAijcVNFPCAPrep
):
    """Fit compressed VNF OH plus LSF-projected individual-line residual PCA."""

    line_amplitude_pca_asset = VNF_COEFFICIENT_LINE_AMPLITUDE_PCA_ASSET

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_line_amplitude_pca_components: int | None = None,
        **kwargs: Any,
    ) -> None:
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(
            root,
            "vnf_coefficient_line_amplitude_pca_contract",
            "VNF coefficient line-amplitude PCA",
        )
        basis_wave, line_names, line_wave, line_group, mean, components, metadata = (
            _load_line_amplitude_basis(
                str(root / self.line_amplitude_pca_asset)
            )
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError(
                "VNF-coefficient line-amplitude PCA requires its exact native grid"
            )
        selected = int(metadata.get("selected_components", components.shape[0]))
        count = (
            selected
            if n_line_amplitude_pca_components is None
            else int(n_line_amplitude_pca_components)
        )
        if count < 1 or count > components.shape[0]:
            raise ValueError(
                "n_line_amplitude_pca_components must be between "
                f"1 and {components.shape[0]}"
            )
        self.n_line_amplitude_pca_components = count
        self.line_amplitude_pca_metadata = metadata
        self._line_amplitude_names = line_names
        self._line_amplitude_wave = line_wave
        self._line_amplitude_group = line_group
        self._line_amplitude_mean = mean
        self._line_amplitude_components = components[:count]
        super().__init__(wave, *args, **kwargs)
        if not np.array_equal(_individual_line_names(self), line_names):
            raise ValueError("The final individual-line PCA line list changed")
        if not np.array_equal(self._line_wave, line_wave) or not np.array_equal(
            self._line_group,
            line_group,
        ):
            raise ValueError("The final individual-line PCA catalog order changed")

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ):
        started = time.perf_counter()
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        result = super().fit(flux, ivar, verbose=verbose)
        return _joint_line_amplitude_pca_fit(
            self,
            result,
            flux,
            ivar,
            started,
            "vnf_pca_line_amplitude_pca",
            3,
        )


class SkyDecompTelluricCorrectedLinesPalaceAijcVNFLineAdjointPCA(
    SkyDecompTelluricCorrectedLinesPalaceAijc
):
    """Fit PALACE-Aijc VNF OH plus direct line-adjoint residual PCA."""

    line_amplitude_pca_asset = VNF_LINE_ADJOINT_PCA_ASSET

    def __init__(
        self,
        wave: np.ndarray,
        *args: Any,
        n_line_adjoint_pca_components: int | None = None,
        **kwargs: Any,
    ) -> None:
        root = Path(kwargs.get("base_dir") or DEFAULT_DATA_ROOT).resolve()
        _validate_method_asset(
            root,
            "vnf_line_adjoint_pca_contract",
            "VNF line-adjoint PCA",
        )
        basis_wave, line_names, line_wave, line_group, mean, components, metadata = (
            _load_line_amplitude_basis(str(root / self.line_amplitude_pca_asset))
        )
        native_wave = np.asarray(wave, dtype=np.float64)
        if not np.array_equal(native_wave, basis_wave):
            raise ValueError("VNF line-adjoint PCA requires its exact native grid")
        selected = int(metadata.get("selected_components", components.shape[0]))
        count = (
            selected
            if n_line_adjoint_pca_components is None
            else int(n_line_adjoint_pca_components)
        )
        if count < 1 or count > components.shape[0]:
            raise ValueError(
                "n_line_adjoint_pca_components must be between "
                f"1 and {components.shape[0]}"
            )
        self.n_line_amplitude_pca_components = count
        self.line_amplitude_pca_metadata = metadata
        self._line_amplitude_names = line_names
        self._line_amplitude_wave = line_wave
        self._line_amplitude_group = line_group
        self._line_amplitude_mean = mean
        self._line_amplitude_components = components[:count]
        super().__init__(wave, *args, **kwargs)
        if not np.array_equal(_individual_line_names(self), line_names):
            raise ValueError("The VNF line-adjoint PCA line list changed")
        if not np.array_equal(self._line_wave, line_wave) or not np.array_equal(
            self._line_group, line_group
        ):
            raise ValueError("The VNF line-adjoint PCA catalog order changed")

    def fit(
        self,
        flux: np.ndarray,
        ivar: np.ndarray,
        *,
        verbose: bool = False,
    ):
        started = time.perf_counter()
        flux = np.asarray(flux, dtype=float)
        ivar = np.asarray(ivar, dtype=float)
        result = super().fit(flux, ivar, verbose=verbose)
        return _joint_line_amplitude_pca_fit(
            self,
            result,
            flux,
            ivar,
            started,
            "vnf_line_adjoint_pca",
            3,
            design_prefix="LineAdjointPCA",
            component_key="line_adjoint_pca",
        )


__all__ = [
    "LINE_AMPLITUDE_PCA_ASSET",
    "LINE_AMPLITUDE_PCA_FIT_MODEL",
    "PALACE_AIJC_VN_FIT_MODEL",
    "RESIDUAL_PCA_ASSET",
    "RESIDUAL_PCA_FIT_MODEL",
    "VN_LINE_AMPLITUDE_PCA_ASSET",
    "VN_LINE_AMPLITUDE_PCA_FIT_MODEL",
    "VNF_COEFFICIENT_LINE_AMPLITUDE_PCA_ASSET",
    "VNF_COEFFICIENT_LINE_AMPLITUDE_PCA_FIT_MODEL",
    "VNF_LINE_ADJOINT_PCA_ASSET",
    "VNF_LINE_ADJOINT_PCA_FIT_MODEL",
    "NIV_VNF_LINE_AMPLITUDE_PCA_ASSET",
    "SPLIT_ZODI_VNF_LINE_AMPLITUDE_PCA_ASSET",
    "VNF_COEFFICIENT_PCA_PREP_ASSET",
    "VNF_COEFFICIENT_PCA_PREP_FIT_MODEL",
    "SkyDecompPalaceAijcVNFLineAmplitudePCA",
    "SkyDecompPalaceAijcVNFNivContinuumLSFSpline2D",
    "SkyDecompPalaceAijcVNFNivContinuumLineAmplitudePCA",
    "SkyDecompPalaceCorrAijcVNFSplitZodiLSFSpline2D",
    "SkyDecompPalaceAijcVNFSplitZodiLSFSpline2D",
    "SkyDecompPalaceAijcVNFSplitZodiLineAmplitudePCA30",
    "SkyDecompTelluricCorrectedLinesLineAmplitudePCA",
    "SkyDecompTelluricCorrectedLinesPalaceAijc",
    "SkyDecompTelluricCorrectedLinesPalaceAijcVN",
    "SkyDecompTelluricCorrectedLinesPalaceAijcVNFPCAPrep",
    "SkyDecompTelluricCorrectedLinesResidualPCA",
    "SkyDecompTelluricCorrectedLinesVNFPCALineAmplitudePCA",
    "SkyDecompTelluricCorrectedLinesPalaceAijcVNFLineAdjointPCA",
    "SkyDecompTelluricCorrectedLinesVNLineAmplitudePCA",
]
