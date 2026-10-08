"""Ablation configs for the post-reversal-fix simplification pass (2026-09-04).

The decomposition corpus was regenerated with the moon/zodi identifiability
priors (see ``moon_zodi_split_priors_2026-09-03.md``).  Several ML components
were added while the decomposition was reversing the moon and zodi roles on
~98% of moon-up rows, and may have been compensating for that.  Each entry
below is a config override to A/B against the deployed default.

Usage in the training-config cell::

    from mlp_predictor import ablations
    train_cfg.update(ablations.ABLATIONS["A1_no_coupling"])   # or omit for the default
    print(ablations.describe("A1_no_coupling"))

Run one ablation per training run, 10 seeds, same split.

READING THE RESULT -- the aggregate alone is not enough
-------------------------------------------------------
On this corpus the seed-to-seed ``mean_eRMSE`` std is **0.331** and the
10-seed ensemble stderr is **0.105** (~1.3% of the 8.289 mean).  So a change
in the aggregate below ~0.33 is seed noise, not signal.

Judge each ablation on the group-level diagnostics, which is where a
reversal-compensating component would show up:

* ``naive_baseline`` (cell 22) per-group ML gain -- the cross-corpus-safe
  metric.  Deployed reference: moon +23.0%, zodi +41.0%, continuum +10.6%,
  mesospheric +3.9%, ionospheric +5.6%, atomic +20.1%; group-equal +5.5%.
* ``sky_arm_zodi_bias`` (cell 27) -- deployed reference: 0 slices carry an
  ML-specific zodi bias.  Any ablation that puts a slice back over threshold
  is a real regression.
* ``wavelength_residual_atlas`` (cell 39) per-band RMS|frac| -- deployed
  reference: blue 0.59%, mid 1.11%, NIR 0.90%.
* ``sky_arm_disagreement_floor`` (cell 38) err/delta -- compare WITHIN this
  corpus only.  Do not compare against the pre-fix numbers in the changelog:
  the near-far moon disagreement fell 12x when the reversal was removed
  (p50 delta 0.0554 -> 0.0045), so the denominator changed and the ratio is
  not commensurable across corpora.
"""

from __future__ import annotations

#: Ablations that can no longer be run, and why.  Their findings are kept in
#: ``_NOTES`` because they are still the reason the code looks the way it does.
#: ``moon_zodi_coupling_enabled`` was removed in the 2026-09-04 checkpoint
#: refactor after A1 was rejected, so A1 and A5 have no off-switch to flip;
#: ``apply`` would raise on them.  Restoring either means restoring the switch.
RETIRED: dict[str, str] = {
    "A1_no_coupling": "moon_zodi_coupling_enabled removed 2026-09-04",
    "A5_no_coupling_thin_zodi": "depends on A1's switch",
    # 2026-09-05 cleanup: measured, lost, and the knobs they drove were removed
    # so only live code remains.  Findings stay in _NOTES because they are the
    # reason those knobs no longer exist -- collectively they are the record
    # that the moon tail is NOT an architecture problem.  Restoring any of
    # these means restoring its knob.
    "S1a_alpha_arm_ctx": "alpha_arm_ctx removed 2026-09-05 (measured: worse)",
    "S1b_alpha_moon_geom": "alpha_arm_ctx removed 2026-09-05 (measured: worse)",
    "S1c_alpha_mlp": "alpha_arm_ctx/alpha_hidden_dims removed 2026-09-05",
    "C4_cont_amp_only5": "flux-MSE/amplitude decoupling removed 2026-09-05",
    "C5_cont_amp_only20": "flux-MSE/amplitude decoupling removed 2026-09-05",
    "G1_gain_moon": "blend gain removed 2026-09-05 (measured: worse)",
    "G2_gain_moon_zodi": "blend gain removed 2026-09-05 (measured: worse)",
    "G3_gain_all3": "blend gain removed 2026-09-05 (measured: worse)",
    "W1_rowsig_p06": "flux_row_weight_power removed 2026-09-05 (measured: worse)",
    "W2_rowsig_p12": "flux_row_weight_power removed 2026-09-05 (measured: worse)",
    "W3_rowsig_p20": "flux_row_weight_power removed 2026-09-05 (measured: worse)",
}

ABLATIONS: dict[str, dict] = {
    # ---------------------------------------------------------------- A2
    "A2_moon_weight_down": {
        "moon_group_weight": 1.5,
    },
    # ---------------------------------------------------------------- A3
    "A3_no_flux_mse": {
        "flux_mse_groups": (),
    },
    # ---------------------------------------------------------------- A4
    "A4_thin_zodi_head": {
        "zodi_head_extra_dims": (),
    },
    # ---------------------------------------------------------------- B1/B2
    "B1_flux_moon_only": {
        "flux_mse_groups": ("moon",),
    },
    "B2_flux_zodi_only": {
        "flux_mse_groups": ("zodi",),
    },
    # ================================================================
    # S1 -- give the blend alpha the moon geometry it structurally lacks.
    # ================================================================
    # Control: the new features WITHOUT arm context.  Separates "alpha needed
    # moon separation" from "alpha needed to see the arms differ".
    "S1d_moon_geom_sci_only": {
        "alpha_ctx_features": ("moon_up_smooth", "ecl_beta_deg", "airmass",
                               "moon_sep", "moon_signal_proxy"),
    },
    # ================================================================
    # S2 -- additive log-amplitude term in flux space, own lambda.
    # ================================================================
    "S2a_flux_amp_1": {"flux_amp_lambda": 1.0},
    "S2b_flux_amp_5": {"flux_amp_lambda": 5.0},
    "S2c_flux_amp_20": {"flux_amp_lambda": 20.0},
    # Moon-only lambda.  A global lambda helped the moon and monotonically
    # wrecked the zodi, whose amplitude is already pinned by the absolute
    # Leinert anchor -- see the S2 note.  Larger values than the global sweep
    # used, because the log term is scale-free and has to outweigh the ABSOLUTE
    # per-pixel term to move the bright-moon tail at all.
    # MEASURED BEST of the S1/S2 pass, and REPLICATED on a second seed block.
    # See the S2 note for what did and did not hold up.
    "S2d_flux_amp_moon_5": {"flux_amp_lambda": {"moon": 5.0, "zodi": 0.0}},
    "S2e_flux_amp_moon_20": {"flux_amp_lambda": {"moon": 20.0, "zodi": 0.0}},
    "S2f_flux_amp_moon_50": {"flux_amp_lambda": {"moon": 50.0, "zodi": 0.0}},
    # S1's only variant that improved anything (moon geometry WITHOUT arm
    # context) combined with the moon-only amplitude term.
    # S1d + the winning moon-only lambda.  S12 used lambda 20, which alone
    # costs the moon 6pp, so it confounded the combination; this is the pairing
    # of the two individually-best settings.
    "S13_moon_geom_amp5": {
        "alpha_ctx_features": ("moon_up_smooth", "ecl_beta_deg", "airmass",
                               "moon_sep", "moon_signal_proxy"),
        "flux_amp_lambda": {"moon": 5.0, "zodi": 0.0},
    },
    # ================================================================
    # G -- multiplicative blend gain.  See the note for the measurement.
    # ================================================================
    # ================================================================
    # N -- re-confirm the adopted settings on the new-oh-3 corpus.
    # The anchor recentring moved the targets (moon-up: zodi x1.600, moon
    # x0.936, continuum x0.922) and the moon's gain over copy-near fell from
    # +24.0% to +15.7%, so the moon lambda tuned on new-oh-2 has to be re-fit
    # rather than assumed.  Everything else improved.
    # ================================================================
    "N0_no_moon_amp": {"flux_amp_lambda": {"moon": 0.0, "zodi": 0.0}},
    "N2_moon_amp2": {"flux_amp_lambda": {"moon": 2.0, "zodi": 0.0}},
    "N10_moon_amp10": {"flux_amp_lambda": {"moon": 10.0, "zodi": 0.0}},
    # Dark-time zodi now has 20% of rows pinned at the anchor FLOOR (was 0%),
    # so the "zodi amplitude is synthetic" argument is weaker there than it was
    # -- though moon-up is still 86.5% pinned.  Probe a small lambda.
    "NZ_zodi_amp1": {"flux_amp_lambda": {"moon": 5.0, "zodi": 1.0}},
    # Does C1 still earn its place on the new targets?
    "NC_no_continuum_flux": {"flux_mse_groups": ("moon", "zodi")},
    # ================================================================
    # C -- give the diffuse continuum a flux-space term at last.
    # ================================================================
    "C1_flux_continuum": {
        "flux_mse_groups": ("moon", "zodi", "continuum"),
    },
    "C2_flux_cont_amp5": {
        "flux_mse_groups": ("moon", "zodi", "continuum"),
        "flux_amp_lambda": {"moon": 5.0, "zodi": 0.0, "continuum": 5.0},
    },
    "C3_flux_cont_amp20": {
        "flux_mse_groups": ("moon", "zodi", "continuum"),
        "flux_amp_lambda": {"moon": 5.0, "zodi": 0.0, "continuum": 20.0},
    },
    # C4 -- the point of the decoupling: continuum KEEPS its coefficient-space
    # smooth_l1 and gains only the log-amplitude term, which is the part that
    # addresses a scale bias.  C1-C3 switch it to flux MSE instead and cost it
    # its accuracy.
    # ================================================================
    # W -- per-row weighting by the decomposition's own amplitude sigma.
    # ================================================================
    "S12_moon_geom_amp": {
        "alpha_ctx_features": ("moon_up_smooth", "ecl_beta_deg", "airmass",
                               "moon_sep", "moon_signal_proxy"),
        "flux_amp_lambda": {"moon": 20.0, "zodi": 0.0},
    },
}

#: Config keys the deployed notebook still sets but the trainer no longer
#: reads -- it warns about these on every run.  Listed here because two of them
#: are active hazards for this simplification pass: ``moon_zodi_mode`` looks
#: like the coupling's off-switch and is not (hence A1 needed a real one), and
#: ``alpha_ctx_groups`` looks like it restricts ctx-alpha and does not.  An A/B
#: driven through any of these measures nothing at all.
DEAD_CONFIG_KEYS: tuple[str, ...] = (
    "alpha_ctx_groups", "block_cov_loss_groups", "flux_mse_eps_frac",
    "head_extra_dims", "high_airmass_boost", "moon_down_ecliptic_beta_deg",
    "moon_down_ecliptic_boost", "moon_zodi_branch_dims", "moon_zodi_mode",
    "moon_zodi_moon_head_extra_dims", "moon_zodi_zodi_head_extra_dims",
    "relative_mse_eps_frac", "relative_mse_groups", "use_coef_err_weights",
)

_NOTES: dict[str, str] = {
    "A1_no_coupling": """
A1 -- remove the additive moon-zodi coupling (~9k params).  THE PRIME SUSPECT.

  A shared 94->64->32 latent is projected additively into BOTH the moon and the
  zodi head via zero-init linear projectors.  Architecturally that is exactly a
  mechanism for shifting flux between the two families in a correlated way,
  i.e. for undoing a role swap.

  What the trained weights already say: the projectors did NOT stay at zero --
  |W|_F is 0.94 (moon) and 0.25 (zodi), about 10.6% of the moon head's own
  weight norm.  So the coupling is doing something.  But the loading is
  ASYMMETRIC, 3.7x more into moon than zodi, whereas pure reversal
  compensation should be roughly equal-and-opposite.  That is more consistent
  with the stated Phase-F intent -- "give the moon head the restricted
  moon-scatter context it otherwise never sees" -- which would survive the
  corpus fix.  Weight norms cannot settle it; this run can.

  Note the off-switch had to be restored: the no-coupling path was deleted in
  the 2026-08-27 trim, so the changelog's "set the moon_zodi_* entries to None"
  raised a ValueError.  With the same seed the disabled model is now
  BIT-IDENTICAL at initialisation to the enabled one (verified), so the A/B
  isolates the coupling and not a different init.

  Expect if it was reversal compensation: no material change, or an
  improvement.  Expect if it was the context fix: moon gain drops from +23%.
""",
    "A2_moon_weight_down": """
A2 -- lower the moon group weight 3.0 -> 1.5.  THE NEW PROBLEM, not an old one.

  The priors made the moon target strongly heavy-tailed.  Median moon_tot is
  146 against p99 = 1.2e5 (a factor of 830), and the population median moon
  SHARE of the moon+zodi continuum is 0.020 -- the prior floor.  Over half the
  corpus now has essentially no moon, and all the moon signal is concentrated
  in a minority of rows.  Before the fix the moon was spread across every row
  (median coefficient sum 2.14 vs 0.153 now).

  An absolute MSE on such a target is dominated by the few bright-moon rows,
  which is the likely reason new large outliers appeared in the moon panel
  (rows 1183 and 1222 are both moon-dominated, moon share 0.69-0.96).  The
  weight m_moon = 3.0 was tuned when the moon was spread over every row; on a
  target this concentrated the same weight puts proportionally more of the
  gradient on the tail and on the shared trunk.

  Per-row relative MSE would be the better instrument, but ``relative_mse_groups``
  was REMOVED from the trainer in the 2026-08-27 trim -- it is one of the 14
  dead config keys the Trainer now warns about.  Reinstating it is a code
  change, not an ablation; this weight sweep is the config-only probe.

  If lowering the weight improves the moon panel tail without costing the
  moon's +23% baseline gain, the heavy tail is the issue and reinstating
  relative MSE is worth the code change.
""",
    "A3_no_flux_mse": """
A3 -- drop the flux-space MSE.  MEASURED: catastrophic.  Do not revisit.

  RESULT (10 seeds): moon gain over B0_copy_near +24.0% -> +5.5%, zodi
  +41.3% -> +23.2%, and on moon-up rows zodi went NEGATIVE (-11.2%, i.e. worse
  than copying the near arm).  Aggregate mean_eRMSE +0.537 and seed std tripled
  (0.276 -> 0.806).  Continuum -7.1pp and mesospheric -4.5pp too, groups that
  have no flux term at all -- so the effect propagates through the shared
  trunk.  The term is a primary driver of accuracy, not overhead.

  The hypothesis this ablation was built on was WRONG, and the correction
  matters for anyone planning further work here: the term does NOT constrain
  the moon+zodi SUM, so it was never "insurance that survives a role swap".
  Reading ``compressed_loss``: it is strictly per-group and it REPLACES that
  group's coefficient-space smooth_l1 loss (if/else, not an added term).  Each
  group's own coefficients are inverted through its own compressor and basis to
  per-pixel flux and scored there.  A swap would not be invisible to it.

  Consequence for the planned full-spectrum flux loss with its own lambda:
  that is a genuinely different design, not a tuning knob.  Today the flux term
  SUPPLANTS the coefficient loss for moon and zodi; an additive lambda-weighted
  term would keep both signals, and would need the per-group weights re-tuned
  because the current group_loss_weight values were fitted against a loss where
  moon and zodi are scored in flux units and everything else in coefficient
  units.
""",
    "A4_thin_zodi_head": """
A4 -- remove the extra 32-d hidden layer from the isolated zodi head.

  The zodi head is now comfortably the best-behaved group: err/delta = 0.354
  all-regime (the diagnostic's own oracle floor is ~1.41, so it is far inside
  the irreducible band), a +41% gain over the best naive baseline, and zero
  ML-specific bias on all nine slices of the zodi-bias diagnostic.

  A group sitting that far below its floor does not need more capacity -- extra
  capacity there chases irreducible noise and drags the shared trunk, which is
  the caution the disagreement-floor cell prints.  This is the cheap partial
  test.  Removing the isolated zodi BRANCH entirely would need a code switch
  like A1's; do that only if A4 shows the capacity is genuinely unused.
""",
    "B1_flux_moon_only": """
B1/B2 -- give the flux-space loss to ONE group at a time.

  Follow-up to A3, which showed the flux-space term is load-bearing: removing
  it entirely cost moon 18.5pp and zodi 18.1pp of their gain over
  B0_copy_near, took moon-up zodi NEGATIVE (-11.2%, worse than copying the
  near arm), and tripled the seed std (0.28 -> 0.81).

  Note what the term actually is, because it is easy to misread: it is
  strictly PER-GROUP and it REPLACES that group's coefficient-space
  smooth_l1 loss (an if/else in ``compressed_loss``, not an added term).  Each
  group's own coefficients are pushed through its own basis matrix to per-pixel
  flux and scored there.  No moon+zodi SUM is constrained anywhere -- so the
  term is not "insurance on the total", and asking whether the gain comes from
  the sum or from per-coefficient weighting is not a well-posed question.

  What the split does test is SEPARABILITY.  If the mechanism is purely
  per-group, B1 should hold the moon near the default +24% while zodi falls
  back toward A3's +23%, and B2 should mirror that.  If instead both groups
  degrade in both arms, the effect is routed through the shared trunk -- which
  A3 already hints at, since it also cost mesospheric 4.5pp and continuum
  7.1pp, groups that have no flux term at all.

  Only ``moon`` and ``zodi`` have basis matrices
  (``_precompute_flux_basis_and_geometry``), so adding other groups to
  ``flux_mse_groups`` silently skips them -- supplying more bases is part of
  the full-spectrum flux-loss work, not a config change.
""",
    "S1a_alpha_arm_ctx": """
S1a/S1b/S1c/S1d -- the blend alpha is the pure-AMPLITUDE knob, and it is blind.

  WHY: measured on the current corpus, 83% of the moon tail's MSE and 89% of
  the zodi tail's is removed by a single per-row rescale -- the network has the
  spline COLOUR right and the BRIGHTNESS wrong, across a 16x amplitude range.
  alpha interpolates the two arms' scores and cannot change their colour, so it
  is exactly the amplitude channel.

  WHAT IS BLIND: ``alpha = sigmoid(Linear(sci_ctx[airmass, ecl_beta_deg,
  moon_up_smooth]))``.  Two independent defects:
    * it reads only ``sci_ctx``, so it cannot express "the science field
      resembles the near arm more than the far one";
    * of its three features, ``moon_up_smooth`` is IDENTICAL across the three
      pointings (it is a property of the moon, not of the pointing), so even
      with arm context it carries no arm information.  Measured median
      |near - far|, as a fraction of each feature's own spread: moon_sep 0.88,
      ecl_beta_deg 1.01, airmass 0.65, moon_up_smooth 0.00, moon_alt 0.00,
      moon_fli 0.00.
  ``moon_sep`` -- the most arm-discriminating feature there is, median
  |near - far| = 22.7 deg, and the one that governs scattered moonlight -- is
  absent from the list entirely.

  The alpha predictors ARE learning (all 10 seeds agree in sign, bias -> alpha0
  ~ 0.90 for moon), so this is a missing-input problem, not a dead-parameter
  one.  Note the per-group "delta alpha at best epoch" the trainer prints is
  uninformative for moon/zodi/continuum: it reports the unused
  ``blend_alpha_direct`` scalar, not the ctx predictor actually in use.

  S1a isolates arm context, S1b adds the moon-geometry features, S1c allows
  depth (arm brightness is not linear in separation), S1d is the control that
  adds the features WITHOUT arm context.  S1d vs S1b attributes the effect.

  MEASURED (10 seeds, 2026-09-04): the hypothesis was WRONG and the CONTROL
  won.  Arm context made the moon flux tail worse in all three variants that
  used it (S1a +4.1%, S1b +3.9%, S1c +5.1%) and more capacity made the moon
  coefficient gain worse (+23.2% -> +19.0% -> +18.0% for S1a/b/c vs +21.6%
  default).  S1d -- the same moon-geometry features read from sci_ctx ALONE --
  was the only variant to improve anything: moon gain +23.9%, moon tail -1.4%,
  sum tail -0.9%, zodi tail -4.1%.  Even that is within seed noise on the tail.
  Reading: alpha wanted moon SEPARATION, not arm context.  Arm context triples
  the input dim of a parameter that already needs alpha_lr_mult=30 to learn at
  all, and 10 extra weights per group behind a sigmoid near 0.9 cost more than
  the asymmetry buys.  Do not retry arm context without first making alpha
  easier to train.

  THE CEILING, MEASURED -- and it is the reason S1 AND S2 both failed on the
  tail.  alpha is a CONVEX blend, so every blended score lies BETWEEN the arms.
  Asking whether the true integrated moon amplitude is reachable that way:

                     inside arm bracket   median miss (arm spans)   p90
      all rows              37.7%                  0.51x           4.2x
      core                  38.7%                  0.46x           3.5x
      tail                  19.6%                  2.18x          17.3x

  On 80% of tail rows the truth is OUTSIDE the interval the two arms span, and
  nominally more than TWO arm-spans outside.  On tail rows the science pointing
  is also inside the arm bracket only 28.6% of the time in moon_sep and 23.2%
  in airmass and alt, against ~47% in the core.

  DO NOT read that as "the sky telescopes do not sample the science field" --
  that was the first conclusion drawn here and it is WRONG.  The arm-span unit
  is misleading precisely on tail rows, because there the two arms AGREE, so
  the span is narrow and a large multiple of it is still a small relative
  distance.  Measured directly: log10(moon_true(sci)/moon_true(near)) on tail
  rows has MAD 0.010 dex, i.e. the true science moon amplitude is within ~2.3%
  of the near arm's.  There is no large extrapolation to perform.

  A geometry transfer ratio was tested on the strength of the wrong reading and
  is REFUTED: log10(true ratio / predicted ratio), MAD, on tail rows --
  physical-model ratio 0.019, crude proxy 0.035, and simply assuming ratio = 1
  gives 0.010.  The model ratio is WORSE than assuming no transfer at all, even
  though the flux-scale calibration cancels in it.  Do not revisit.

  What remains true is only the negative result: alpha's convexity plus its
  blindness to moon geometry does not explain the tail, and neither a smarter
  alpha nor a scale-free amplitude loss moves it.  Note also that "amplitude"
  has been used for two different quantities here -- the best-rescale factor
  s* (flux-weighted, p10-p90 0.90-1.24 on the tail) and the plain integrated
  flux (median relative error 3.4% on the tail).  The 83% amp-share figure is
  about the FORMER.  Any further work on the tail should first establish which
  of the two is actually wrong, because they disagree by a factor of ~4.
""",
    "S2a_flux_amp_1": """
S2 -- additive log-amplitude term in flux space, tuned by ``flux_amp_lambda``.

  WHY: the existing flux-space term is an ABSOLUTE per-pixel MSE, so it is
  dominated by the brightest rows and nearly blind to a 20% brightness miss on
  a faint one.  But the tail error IS brightness (83-89% of its MSE), and it
  spans a 16x amplitude range between core and tail.  This term is scale-free:
  ``log((A_pred + eps)/(A_true + eps))**2`` on the wavelength-integrated flux
  penalises the same fractional miss equally at every brightness.

  ADDITIVE, deliberately.  A3 measured what happens when the per-pixel term is
  removed instead of augmented: moon gain +24.0% -> +5.5%, zodi +41.3% ->
  +23.2%, moon-up zodi NEGATIVE, seed std tripled.  So this adds to that term
  and does not replace it.

  eps is 5% of the group's median train amplitude (``flux_amp_floor_frac``), so
  rows far below the typical brightness -- dark-time moon, which is at the
  decomposition's 2% share floor and is spurious anyway -- contribute ~0 rather
  than dominating a log ratio.

  SCALE: the per-pixel term is multiplied by ``scale_match`` to sit at the
  coefficient loss's magnitude, while ``log**2`` is naturally O(0.01) for a
  0.1 dex miss.  So lambda has to be O(1-10) to matter at all; hence the 1/5/20
  sweep rather than a fine one.  If lambda 20 helps and 5 does not, sweep
  higher before concluding.

  MEASURED (10 seeds, 2026-09-04), global lambda over moon AND zodi:
    moon  median relative integrated-amplitude error 0.0774 -> 0.0615 (lambda 1)
          / 0.0613 (lambda 5) -- a 21% reduction, exactly what it was built for.
    zodi  monotonic damage: gain over copy-near +38.3% -> +35.7% / +29.0% /
          +21.8% for lambda 1/5/20, tail +6.7% / +17.7% / +25.3%, amp share
          0.877 -> 0.765.  That is the "amplitude improved, colour degraded"
          signature.
  CAUSE: the zodi amplitude is already synthetic.  93% of moon-up rows sit
  exactly on the decomposition's absolute Leinert ceiling, so the integrated
  zodi is a deterministic function of geometry the network already predicts
  nearly perfectly; penalising it harder buys nothing and is paid in colour.
  Hence ``flux_amp_lambda`` now accepts a per-group dict -- see S2d/S2e/S2f.

  The moon TAIL did not move either (0.2545 -> 0.2531 at lambda 5).  Consistent:
  the log term is scale-free while the per-pixel term is absolute, so on the
  bright rows that form the tail the absolute term still dominates the gradient.
  Moving the tail needs a lambda large enough to compete there, which the global
  sweep could not reach because the zodi collapsed first.

  MOON-ONLY lambda (S2d/S2e/S2f), 10 seeds, and REPLICATED on seeds 52-61
  because the first block flattered it.  What held up in BOTH blocks:
    moon median relative integrated-amplitude error  -20.3% / -18.6%  <- the
      mechanism the term exists for; this is the result.
    moon core flux RMSE   -2.3% / -1.4%
    zodi core flux RMSE   -3.0% / -1.4%
    continuum gain        +1.4pp / +1.6pp
    mesospheric gain      -0.6pp / -0.8pp   <- small consistent COST
    mean_eRMSE            +0.054 / +0.090   <- consistently slightly WORSE,
      because that aggregate is dominated by the 358 mesospheric coefficients
      (scale ~34) not by moon and zodi (scale ~0.5).
  What did NOT hold up, and was over-read on the first block alone:
    seed std      0.313 -> 0.170 looked like a halving; on seeds 52-61 it was
      0.378 -> 0.408, i.e. no variance reduction at all.  A seed-std estimate
      from 10 seeds carries ~24% of its own uncertainty; do not read one.
    moon coef gain  +2.6pp -> only +0.7pp in the second block.
    zodi tail       -7.2% -> -0.5%.  Noise.
  lambda ESCALATION is harmful, refuting the idea that the tail just needed a
  bigger lambda: moon-only lambda 5/20/50 gives moon gain +24.2% / +18.2% /
  +9.7%.  lambda ~5 is near-optimal.

  THE MOON TAIL DOES NOT MOVE at any lambda (+0.3% / -0.1% across blocks), and
  that is now explained rather than open -- see the convexity ceiling in the S1
  note.  Loss reweighting cannot fix a limit on what the architecture can
  represent.

  WATCH: ``group_loss_weight`` was tuned against a loss where moon and zodi are
  scored in flux units.  A large lambda changes that balance, so a win here may
  need moon_group_weight re-tuned jointly -- check the moon/zodi gains did not
  simply trade against continuum and mesospheric.
""",
    "G1_gain_moon": """
G1/G2/G3 -- a multiplicative gain on the arm blend.

  WHY: the tail error is a per-row MULTIPLICATIVE brightness error, and no knob
  in the architecture expressed one.  alpha mixes the arms, so it can only
  reach values BETWEEN them; the head corrects with an additive OFFSET.  Since
  the compressor is sqrt, a factor k in flux is a factor sqrt(k) in score space
  -- still multiplicative, so an additive offset is the wrong shape for it.

  WHAT THE MEASUREMENTS SAY (new-oh-3, tail = worst 5% by absolute flux RMSE):
    alpha leverage |n-f|/max(n,f)   moon 11.4%   zodi 44.6%
    amplitude error to fix          moon  7.4%   zodi 34.6%
    error / leverage                moon  0.65   zodi  0.77
    truth outside the arm span by   moon  6.8%   zodi 21.5%
    rows with truth inside the span      26.3%        26.3%
  So alpha is NOT out of range -- it has 1.3-1.5x the authority needed, which
  refutes the earlier "alpha structurally cannot reach it" claim.  But the truth
  sits outside the arm span on ~74% of tail rows, and the model's error (7.4%)
  is close to the ~6.8% floor that the best CONVEX alpha would leave, i.e. the
  additive head is not doing much extrapolating in practice.

  NOT A REACHABILITY FIX.  ``out[g] = blend + head`` with an unbounded final
  Linear, so the model could always represent any amplitude.  This changes the
  PARAMETRISATION, not the expressible set.  If it fails, that is evidence the
  parametrisation was never the obstacle -- do not respond by widening alpha's
  range, which the leverage numbers already rule out.

  gain = exp(max_log * tanh(z)), final layer zero-init, so gain == 1.0 exactly
  at init and the enabled model starts BIT-IDENTICAL to the disabled one.  It
  joins alpha in the wd=0, alpha_lr_mult parameter group: it is the same kind of
  small slow multiplicative parameter, and alpha needed lr x30 before it learned
  at all ([[blend-alpha-frozen-unless-lr-boosted]]).

  READ IT ON: moon/zodi rel-amp (its target), then per-group gain and bias.
  Watch mesospheric -- everything routed through the trunk has leaked before.
""",
    "W1_rowsig_p06": """
W1/W2/W3 -- per-row weighting of the flux groups by the decomposition's own
amplitude sigma.  MEASURED WORSE, monotonically.  Do not revisit without a
different statistic.

  THE IDEA: the integrated amplitude is a linear functional of the
  coefficients, A = c . v with v = basis.sum(axis=1), so the persisted per-row
  QP covariance propagates exactly, Var(A) = v^T Sigma v.  That sigma is the
  ONLY quantity found that correlates with the moon amplitude error --
  rho(log sigma_A, log|err|) = +0.78 on gaia-stars, against R^2 = -0.007 for
  all 37 observing-geometry features.  Weight w_r ~ (sigma_r/median)^-p,
  normalised to mean 1 over train rows, clipped to [0.1, 10].

  RESULT (10 seeds each, gaia-stars), gain over B0_copy_near:

  | p   | moon   | zodi   | moon core | moon tail |
  |-----|--------|--------|-----------|-----------|
  | 0   | +21.3% | +30.4% |  --       |  --       |
  | 0.6 | +18.7% | +26.9% | +7.4% worse | +5.0% worse |
  | 1.2 | +18.0% | +24.1% | +6.9% worse | +6.0% worse |
  | 2.0 | +17.7% | +22.9% | +9.0% worse | +4.3% worse |

  Monotone in p, and worse on the CORE rows too -- so this is not the
  "down-weighting hard rows flatters the aggregate" artifact, it is a real loss.

  WHY, and it is the lesson: a sigma that correlates with |error| can mean
  either NOISE (down-weight and you stop fitting garbage) or DIFFICULTY
  (down-weight and you discard your most informative examples).  rho = 0.78 is
  consistent with both; this result says it is DIFFICULTY.  The notebook's own
  calibration cells said as much in advance and were not read that way: a sigma
  uncalibrated by 50-180x (`sigma_scale hint 0.02x`, truth-conditioned
  median|z| 0.0037 against a target of 0.6745, still ~5x off at SNR>3) with a
  reliability slope of 0.53-0.65 instead of 1 is not measuring a noise level.
  The slope was used to set the exponent (2s ~ 1.2 rather than the naive 2) but
  should also have been read as evidence about what sigma IS.

  Consequence: sigma_A and the ensemble spread (rho = +0.88) are usable for
  FLAGGING rows downstream, not for correcting or reweighting them.  That is
  the same magnitude-only asymmetry every other attack on this error has hit --
  see the S1/G notes and [[moon-zodi-tail-is-amplitude]].

  The helper that computed it was ~25 lines in data.py and was removed with the
  config knob; it is one einsum, `np.einsum('i,rij,j->r', v, C, v)` over
  COEF_COV_MOON / COEF_COV_ZODI indexed [row, k, k].
""",
    "C1_flux_continuum": """
C1/C2/C3 -- a flux-space term for the diffuse continuum (HO2 + FeO + O2Ac).

  WHY: the continuum carries a large SYSTEMATIC that RMSE never showed.
  Signed flux bias on the new-oh-2 test split: continuum -3.8%, against moon
  +0.10% and zodi +0.22%.  In the blue it is -4.1%, and the wavelength atlas
  makes continuum the dominant residual component there (109% of the band's
  total residual RMS).

  MECHANISM: continuum had no flux-space term, so it was fit purely in
  compressed coefficient space, and the trainer's empirical calibration
  corrects its MEAN COEFFICIENT (a +1.54% lift) -- which is not its
  flux-weighted bias, because HO2, FeO and O2Ac have very different flux
  integrals.  A group can look calibrated per-coefficient and be 4% low in flux.

  This was NOT a design decision.  ``matrix_diffuse`` has existed all along in
  the same (n_coef, n_wave) layout as moon and zodi; it was simply never added
  to ``_precompute_flux_basis_and_geometry``, so naming ``continuum`` in
  ``flux_mse_groups`` silently skipped it.  Earlier notes in this file claiming
  only moon and zodi have bases were wrong.

  READ THE RESULT ON BIAS, NOT RMSE.  The harness reports signed flux bias per
  group for exactly this reason.  Also remember the flux term REPLACES that
  group's coefficient-space smooth_l1 rather than adding to it, so C1 is a
  switch of loss space for continuum, not an extra term -- and A3 measured that
  removing the coefficient loss from a group is not automatically safe.

  C2/C3 add the log-amplitude term on continuum as well, which is the part that
  penalises a scale error directly.  lambda 5 mirrors the adopted moon setting;
  20 tests whether the bias needs more force.  Watch mesospheric and the moon:
  continuum shares the trunk with them, and A3 showed effects propagate there.
""",
    "A5_no_coupling_thin_zodi": """
A5 -- A1 + A4 together.

  Run only after A1 and A4 have each been measured alone.  Both touch the
  moon/zodi pair, so a combined run cannot attribute a change to either, but if
  both are individually neutral this confirms they are neutral together rather
  than compensating for one another.
""",
}


def describe(name: str) -> str:
    """Rationale, prior evidence, and what to watch for one ablation."""
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; have {sorted(ABLATIONS)}")
    return _NOTES.get(name, "(no notes)").strip("\n")


def apply(train_cfg: dict, name: str, verbose: bool = True) -> dict:
    """Apply one ablation's overrides in place and report what changed."""
    if name in RETIRED:
        raise KeyError(
            f"ablation {name!r} is retired ({RETIRED[name]}); its findings are "
            f"in describe({name!r}) but it cannot be run as-is.")
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; have {sorted(ABLATIONS)}")
    overrides = ABLATIONS[name]
    # Validate against the trainer's own consumed-key set, not just against
    # train_cfg: a key can be present in the config and still be dead (14 of
    # them currently are).  Setting a dead knob would silently produce a null
    # A/B, which is exactly the failure this pass exists to avoid.
    from . import trainer as _trainer
    live = getattr(_trainer.Trainer, "_CONSUMED_CFG_KEYS", None)
    for key in overrides:
        if key not in train_cfg:
            raise KeyError(
                f"ablation {name!r} sets {key!r}, which is not in train_cfg; "
                "the config schema has changed and this ablation needs updating."
            )
        if live is not None and key not in live:
            raise KeyError(
                f"ablation {name!r} sets {key!r}, which the trainer does NOT read "
                f"(see ablations.DEAD_CONFIG_KEYS). The run would be identical to "
                f"the default -- fix the ablation or plumb the knob through."
            )
    if verbose:
        print(f"=== ablation {name} ===")
        for key, new in overrides.items():
            print(f"  {key}: {train_cfg[key]!r} -> {new!r}")
    train_cfg.update(overrides)
    return train_cfg


__all__ = ["ABLATIONS", "RETIRED", "DEAD_CONFIG_KEYS", "describe", "apply"]
