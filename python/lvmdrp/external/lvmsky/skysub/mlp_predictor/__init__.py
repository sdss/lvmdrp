"""Same-time sky→science coefficient predictor.

Reorganisation of the split-zodi spline4 notebook into a package.
Modules
-------
- ``config``      : dataclasses for pipeline configuration.
- ``ml_utils``    : RobustScaler, seed / split helpers.
- ``metrics``     : coefficient-space and pixel-space WRMSE helpers.
- ``model``       : ``DualEncoderGroupHeadMLP`` architecture.
- ``compressor``  : per-group coefficient compressor (sqrt + optional PCA).
- ``data``        : ``DataPipeline`` — loads decompositions, builds triplets,
                    filters, augments with ecliptic + physics-prior features.
- ``wavelengths`` : per-coefficient effective wavelength + extinction cache.
- ``losses``      : compressed / flux-MSE loss builders.
- ``trainer``     : ensemble-seed training loop.
- ``diagnostics`` : batch RMSE, worst-recon, sigma-calibration, deployment
                    drift, WRMSE-vs-context plot / stat functions.
"""

from importlib import import_module as _import

__all__ = [
    "config",
    "ml_utils",
    "metrics",
    "model",
    "compressor",
    "data",
    "wavelengths",
    "trainer",
    "diagnostics",
]

for _m in __all__:
    _import(f".{_m}", package=__name__)
