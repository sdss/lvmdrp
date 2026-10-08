# Producer cell: builds the full-triplet per-row sWRMSE_coef and the
# train / val-test / LMC-SMC masks over the FULL row set (no region filter).
# Populates the kernel state consumed by the wrmse_vs_ctx_* correlation and
# scatter cells.  The Galactic-coordinate map plot from the original
# ``manual-fulltriplet-r2-map`` cell is intentionally omitted here — the
# correlation / scatter diagnostics do not need it.
from astropy.coordinates import SkyCoord
import astropy.units as u

required = ['filtered_triplet', 'train_idx', 'val_idx', 'test_idx',
            'mlp_artifacts', 'predict_sci_coefficients_default',
            'build_triplet_coef_dataset']
_missing = [k for k in required if k not in globals()]
if _missing:
    raise RuntimeError(
        'Missing kernel state: ' + ', '.join(_missing)
        + '. Run all prior cells (data loading + model training + split '
        'assignment) before executing this cell.'
    )

_suffix = globals().get('_DECOMP_SUFFIX', '_lsf_surface_iterative_split_zodi')
INPUT_FITS = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}.fits'
NEAR_FITS = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_decomp_sky1{_suffix}.fits'
FAR_FITS = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_decomp_sky2{_suffix}.fits'
SCI_FITS = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_decomp_sci{_suffix}.fits'
META_ONLY_FITS = f'{DECOMP_DATA_ROOT}/{DECOMP_STEM}_meta_only.fits'

with fits.open(META_ONLY_FITS) as hdul:
    meta_tbl_all = Table(hdul['META'].data)

n_all_total = len(meta_tbl_all)
ra_col = next((c for c in ['sci_ra', 'ra', 'RA'] if c in meta_tbl_all.colnames), None)
dec_col = next((c for c in ['sci_dec', 'dec', 'DEC'] if c in meta_tbl_all.colnames), None)
if ra_col is None or dec_col is None:
    raise ValueError('RA/DEC not found in metadata')

ra_all = np.asarray(meta_tbl_all[ra_col], dtype=np.float64)
dec_all = np.asarray(meta_tbl_all[dec_col], dtype=np.float64)
coords_icrs = SkyCoord(ra=ra_all * u.deg, dec=dec_all * u.deg, frame='icrs')

lmc_cfg = globals().get('LMC_EXCLUSION', {'ra_deg': 80.894, 'dec_deg': -69.756, 'radius_deg': 10.0})
smc_cfg = globals().get('SMC_EXCLUSION', {'ra_deg': 13.187, 'dec_deg': -72.829, 'radius_deg': 10.0})

is_region_excluded = np.zeros(n_all_total, dtype=bool)
for _cfg in (lmc_cfg, smc_cfg):
    _center = SkyCoord(ra=float(_cfg['ra_deg']) * u.deg,
                       dec=float(_cfg['dec_deg']) * u.deg, frame='icrs')
    is_region_excluded |= (coords_icrs.separation(_center).deg <= float(_cfg['radius_deg']))

filtered_row_indices = np.asarray(filtered_triplet['row_index'], dtype=int)
is_train = np.zeros(n_all_total, dtype=bool)
is_valtest = np.zeros(n_all_total, dtype=bool)
train_idx_arr = np.asarray(train_idx, dtype=int)
valtest_idx_arr = np.unique(np.concatenate([np.asarray(val_idx, dtype=int),
                                            np.asarray(test_idx, dtype=int)]))
is_train[filtered_row_indices[train_idx_arr]] = True
is_valtest[filtered_row_indices[valtest_idx_arr]] = True
is_other = ~(is_train | is_valtest | is_region_excluded)

print(f'Row counts: total={n_all_total} '
      f'train={is_train.sum()} valtest={is_valtest.sum()} '
      f'lmcsmc={is_region_excluded.sum()} other={is_other.sum()}')

# Full aligned triplet without region filter -> includes LMC/SMC rows.
# Strip ecliptic + physics-prior features from the loader context columns;
# they are attached post-hoc.
context_columns = list(globals().get('ctx_names_all', filtered_triplet['ctx_names']))
_ecl_names = (set(globals().get('ECLIPTIC_FEATURE_NAMES', []))
              | set(globals().get('PHYSICS_PRIOR_FEATURE_NAMES', [])))
context_columns = [c for c in context_columns if c not in _ecl_names]

triplet_full = build_triplet_coef_dataset(
    input_fits_path=INPUT_FITS,
    sky_near_decomp_fits_path=NEAR_FITS,
    sky_far_decomp_fits_path=FAR_FITS,
    sci_decomp_fits_path=SCI_FITS,
    context_columns=context_columns,
    return_chi2=False,
)
if '_augment_triplet_with_ecliptic' in globals():
    _augment_triplet_with_ecliptic(triplet_full, meta_fits_path=INPUT_FITS)
if '_augment_triplet_with_physics_priors' in globals():
    _augment_triplet_with_physics_priors(triplet_full)
    # MOON-MODEL-CTX-V1: must match the training-time ctx layout, or
    # ctx_names will not line up with the trained ensemble.
    if (globals().get('USE_MOON_MODEL_FEATURE', False)
            and '_augment_triplet_with_moon_model' in globals()):
        _augment_triplet_with_moon_model(triplet_full, MOON_MODEL_CORPUS_PREFIX)

full_rows = np.asarray(triplet_full['row_index'], dtype=int)
coef_true_full = np.asarray(triplet_full['coef_sci'], dtype=np.float64)
coef_pred_full = predict_sci_coefficients_default(
    mlp_artifacts,
    coef_near_phys=triplet_full['coef_near'],
    coef_far_phys=triplet_full['coef_far'],
    ctx_near_phys=triplet_full['ctx_near'],
    ctx_far_phys=triplet_full['ctx_far'],
    ctx_sci_phys=triplet_full['ctx_sci'],
).astype(np.float64)

_sigma_full = triplet_full.get('coef_err_sci', None)
if _sigma_full is None:
    _sigma_full = np.full_like(coef_true_full, np.nan)
_sigma_full = np.asarray(_sigma_full, dtype=np.float64)
_gidx_map = (_group_indices_compress if '_group_indices_compress' in globals()
             else group_indices)
wrmse_full = weighted_rmse_per_row(
    coef_true_full, coef_pred_full, _sigma_full,
    _gidx_map, dict(DEFAULT_COEF_ERR_SIGMA_FLOOR_BY_GROUP),
).astype(np.float32)

wrmse_all = np.full(n_all_total, np.nan, dtype=np.float32)
valid_rows = (full_rows >= 0) & (full_rows < n_all_total)
wrmse_all[full_rows[valid_rows]] = wrmse_full[valid_rows]

print(f'Finite sWRMSE_coef: total={np.isfinite(wrmse_all).sum()} '
      f'train={np.isfinite(wrmse_all[is_train]).sum()} '
      f'valtest={np.isfinite(wrmse_all[is_valtest]).sum()} '
      f'lmcsmc={np.isfinite(wrmse_all[is_region_excluded]).sum()} '
      f'other={np.isfinite(wrmse_all[is_other]).sum()}')
