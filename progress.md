# Progress Log — Regional Flood Frequency Analysis (RFFA)

Weekly log for prof updates. Newest week first. Every commit from here on gets a line in the
current week's section; a new week heading is started each Monday.

---

## Week of 2026-10-06

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
