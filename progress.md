# Progress Log — Regional Flood Frequency Analysis (RFFA)

Weekly log for prof updates. Newest week first. Every commit from here on gets a line in the
current week's section; a new week heading is started each Monday.

---

## Week of 2026-10-06 (cont'd, 2026-10-08)

Built Step 3, 4, 5 from scratch against the corrected Step 1/Step 2 outputs only (the old
`stage 2-5.py` scripts are no longer used anywhere in this pipeline).

- **`code/step3.py` — distribution selection + regional growth curve** (Hosking & Wallis
  1997 ch. 5). For each target's final ROI: regional-average L-moments -> fit GLO/GEV/GPA
  (all three are the already-verified 4-parameter kappa distribution at fixed h=-1/0/+1, so
  no new approximation needed) plus GNO/PE3 (fit by numerically integrating L-moments of
  their quantile functions and root-finding the shape parameter) -> score each with the
  Z^DIST statistic (same kappa-Monte-Carlo machinery as roi.py's H1-H3) -> pick the
  best-fitting, |Z|<=1.64 preferred -> growth curve q(T) at T=2,5,10,25,50,100.
  A self-test runs at import time (kappa-exact vs numerical-integration agreement; GNO/PE3
  reduce to the normal distribution at shape=0) so a wrong formula fails loudly, not
  silently. Result: 83/90 regions get an acceptably-fitting distribution (|Z|<=1.64); GLO
  (27), GPA (18), PE3 (16), GNO (15), GEV (14) all get chosen somewhere.
- **`code/step4.py` — multiple linear regression for the index flood (mu)**. Forward
  selection on adjusted R^2 with a VIF<=5 collinearity guard, Duan smearing bias correction
  on the log10 back-transform. Full-sample result: predictors
  {lat, log10(area), rainfall_max_mm, temp_max_C, lon, mean_elev_m}, adjusted R^2 = 0.40,
  median in-sample |error| on mu = 66%. Confirms the Step 2 finding that these attributes
  only weakly predict flood behaviour -- not a bug, a real data limitation.
- **`code/step5.py` — the ungauged-site test, full leave-one-out over all 90 stations**
  (not just one, per "the right way to test"). For each station: mask its own flow record,
  re-derive its region from gauged neighbours only (Di/H1-H3 computed on the pool, never on
  the masked target), refit the Step 4 regression excluding it, predict Q(T) from
  attributes alone, and compare to its real at-site answer.
  - **Region stability** ("does it join the same stations as before?"): median Jaccard
    overlap with the original (non-masked) region = 0.80; exact same neighbour set in
    10/90 cases; zero overlap in none of them -- attribute-based pooling is reasonably
    robust to not having the target's own gauge.
  - **Q(T) error**: median |% error| ~70-77% across all return periods (T=50: 74%, T=100:
    76%), consistent with Step 4's weak R^2 -- errors do not notably worsen with return
    period here, unlike the usual tail-extrapolation expectation, likely because the index
    flood (mu) prediction error dominates over the growth-curve shape error.
  - 2 of 90 stations (`Khanditar`, `T. narasipur`) get no valid at-site distribution fit at
    all -- both are exactly the stations already flagged as discordant against the whole
    dataset (and `Khanditar` is the one with the suspected unit error, 641,030 vs a 3,651
    median). Correct "no fit" rather than a wrong number, not a bug.
  - Outputs: `step5_ungauged_test.csv` (full per-station results), `step5_region_overlap.csv`,
    `step5_est_vs_computed_T50.png` / `_T100.png` (Est-vs-Computed scatter, 1:1 line, as
    asked), `step5_error_by_return_period.png`, `step5_summary.txt`.

- Reviewed `code/step1.py` (attribute finalisation + PCA) and `code/roi.py` (Step 2, Region of
  Influence) end-to-end against Hosking & Wallis (1997) for correctness.
- **Confirmed correct, no change:** PCA is properly standardised — all 10 attributes are z-scored
  (`StandardScaler`) before PCA, with `area_km2` log-transformed first (raw skew 4.2 → 0.09 after
  log). The H1/H2/H3 heterogeneity statistic and the Di discordancy statistic already implement
  the "divide by std dev" standardisation correctly (verified both formulas algebraically against
  the published equations) — H1 is literally `(V_observed - mean(V_simulated)) / std(V_simulated)`
  from the kappa-distribution Monte Carlo null.
- **Bug fixed — attribute redundancy in PCA:** found 3 of the 15 catchment attributes were exact
  algebraic duplicates of others already in the set (`stream_len_km = drain_density * area_km2`
  exactly; `relief_m = max_elev_m - min_elev_m` exactly; `centroid_lat/lon` ≈ `lat/lon` at r=0.998/
  0.994). Carrying both sides of an identity into PCA triple-weights that signal and starves the
  independent ones (slope, rainfall, temperature). Trimmed `ATTRS` in `step1.py` from 15 → 10.
- **Bug fixed — premature ROI growth stopping:** `ROI_PATIENCE` in `roi.py` could trigger off H1
  computed at region size 5-6, which H&W themselves note is too small for H to be a reliable
  statistic. Added `ROI_PATIENCE_GRACE=10` so growth can't be cut short before H1 is a meaningful
  signal (fixed in both `grow()` and the independent counter in `choose()`).
- Re-ran Step 1 + Step 2 with both fixes. Result: acceptably-homogeneous ROIs improved from 18/90
  to 23/90 stations (H1<1); definitely-heterogeneous dropped from 60/90 to 54/90.
- **Open finding, not a bug:** `roi.py`'s own diagnostic shows attribute-space distance is a weak
  predictor of flood L-CV similarity (Spearman rho ≈ 0.05-0.06; 5 nearest PCA-neighbours only ~22%
  tighter in L-CV than 5 random stations). Tested directly by letting stuck regions grow past the
  old size-5 floor — H1 got *worse*, not better, in 68% of growth steps. This means the 10 static
  morphometric/climate attributes genuinely don't separate flood-generating behaviour well for a
  majority of catchments; more growth/patience won't fix it. Will need to be stated as a limitation
  in the write-up, or addressed by adding attributes (land use, soil, geology) not currently in the
  dataset.
- Flagged for manual check (not auto-removed): `Sevanur` and `Tuichang` have |z|>3 on one attribute
  each; `Khanditar`'s streamflow record has one year at 175x its own median (641,030 vs 3,651
  cumec) — worth checking the raw file for a possible unit/digit error.
- Set up git + GitHub remote (`manasareddy181006-rgb/flood-prediction`) for this project.
  Going forward: every change is committed and pushed, and gets a line here, so the log always
  matches what's on GitHub.
- Standing rules set up (no need to ask going forward):
  - Every change gets pushed straight to GitHub, and a line added here.
  - `claude_prompt_history.md` is now auto-appended on every prompt via a `UserPromptSubmit` hook
    (`.claude/settings.json` + `.claude/hooks/log_prompt.py`) so the full prompt history is tracked
    in git for review.
  - An IITM-template slide deck is due at the end of every week (pending: need the actual IITM
    Beamer/PPT template source to match the branding exactly — asked prof/user for it).

---

## Week of 2026-09-22

- Rewrote the attribute-finalisation + PCA step as `code/step1.py` and the pooling step as
  `code/roi.py` (Region of Influence, Hosking & Wallis 1997), replacing the earlier `stage 2-5`
  scripts with one coherent, better-documented pipeline.
- Step 1: merged morphometry (status == SUCCESS), rainfall and temperature attributes; kept
  stations with >15 years of annual-maximum streamflow; ran PCA (all 15 attributes, z-scored, all
  components retained for Step 2).
- Step 2: grew a bespoke Region of Influence for each of 90 target catchments by nearest-neighbour
  distance in PCA space, with discordancy (Di) exclusion and heterogeneity (H1/H2/H3) monitoring
  via kappa-distribution Monte Carlo (Hosking & Wallis 1997 s4.3).
- Outputs: `step1_outputs/`, `step2_outputs/` (distance matrix, nearest-15-neighbours, ROI members,
  growth paths, summary plots).

---

## Week of 2026-09-15

- Built the original `stage 3`, `stage 4`, `stage 5` scripts: within-region discordancy/
  heterogeneity/goodness-of-fit, ROI growth curves + ROI-vs-fixed-clustering comparison, and the
  index-flood regression for ungauged-site quantile estimates.
- Outputs: `rfa_homogeneity.csv`, `rfa_roi_pools.csv`, `rfa_quantiles.csv`, `rfa_validation.csv`,
  `rfa_indexflood_model.txt`, `rfa_final_quantiles.csv`.

---

## Week of 2026-09-08

- Initial setup: morphometry extraction (`code_v2.py`, `check*.py`), catchment mosaicking/cropping
  GIS scripts, and the first PCA + clustering pass (`pca_roi.py`, `Output_PCA_Clustering/`).
- Produced `All_Stations_morphometry_success.csv` and the India catchments overview map.
