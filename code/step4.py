"""
STEP 4 - MULTIPLE LINEAR REGRESSION FOR THE INDEX FLOOD (mu) OF AN UNGAUGED SITE
==================================================================================
Step 3 gives the dimensionless REGIONAL growth curve q(T) (mean = 1) for each target's
ROI. To turn that into a real discharge, Q(T) = mu_hat * q(T), mu_hat (the index flood,
here taken as the at-site mean annual maximum) must come from catchment ATTRIBUTES alone
-- that is what makes the method usable at an ungauged site, and it is the step that
"removes the normalisation" (un-does the /mu done before pooling in Step 3).

METHOD (Hosking & Wallis 1997 s8; standard RFA practice)
  Fit   log10(mu_i) = b0 + b1*log10(area_i) + b2*X2_i + ...   across the gauged sites.
  Log-log in area is standard: flood magnitude scales roughly as a power of catchment
  area, so the relationship is linear in logs with more nearly constant-variance errors.

  Predictors are added by FORWARD SELECTION on adjusted R^2 (only kept if it improves
  fit), each candidate screened by VARIANCE INFLATION FACTOR (VIF <= 5) against predictors
  already in the model, so two attributes that carry the same information (e.g. latitude
  and temperature, r = -0.93 in this dataset) cannot both enter.

  The log-space fit is bias-corrected on back-transformation with Duan's SMEARING
  ESTIMATE (mean of 10**residual across the fitted sample) -- a naive 10**(fitted log)
  systematically under-predicts mu because of Jensen's inequality.

Exposes fit_mlr(attr_df, lm_df, exclude_stations=None) for step5.py's leave-one-out use
(refits the WHOLE forward-selection procedure on the reduced training set each time, not
just a coefficient refit -- otherwise the ungauged test would silently assume the "right"
predictors were already known in advance).

Outputs in <DATA_DIR>/step4_outputs/:
    step4_indexflood_model.txt   full-sample model: predictors, coefficients, diagnostics
    step4_mu_predictions.csv     mu_at_site vs mu_predicted (in-sample) for every station
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
IN1 = os.path.join(DATA_DIR, "step1_outputs")
IN2 = os.path.join(DATA_DIR, "step2_outputs")
OUT_NAME = "step4_outputs"

MAX_VIF = 5.0

# candidate predictors: (output name, source column, transform)
CAND_SPEC = [
    ("log10_area", "area_km2", "log10"),
    ("drain_density", "drain_density", None),
    ("mean_elev_m", "mean_elev_m", None),
    ("relief_m", "relief_m", None),
    ("mean_slope_deg", "mean_slope_deg", None),
    ("lat", "lat", None),
    ("lon", "lon", None),
    ("rainfall_max_mm", "rainfall_max_mm", None),
    ("temp_max_C", "temp_max_C", None),
    ("temp_min_C", "temp_min_C", None),
]


def build_candidates(attr_df):
    X = pd.DataFrame(index=attr_df.index)
    for name, col, tr in CAND_SPEC:
        v = attr_df[col].astype(float)
        X[name] = np.log10(v.clip(lower=1e-6)) if tr == "log10" else v
    return X


# =============================================================================
# OLS + VIF + forward selection
# =============================================================================
def _ols(X, y):
    """X: (n,p) WITHOUT intercept. Returns beta (incl. intercept at index 0), fitted, resid."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    fitted = Xd @ beta
    resid = y - fitted
    return beta, fitted, resid


def _adj_r2(y, resid, p):
    n = len(y)
    sse = np.sum(resid ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    if n - p - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def _vif(X, j):
    """VIF of column j regressed on all other columns in X (ndarray, no intercept)."""
    other = np.delete(X, j, axis=1)
    if other.shape[1] == 0:
        return 1.0
    beta, fitted, resid = _ols(other, X[:, j])
    sse = np.sum(resid ** 2)
    sst = np.sum((X[:, j] - X[:, j].mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    return 1.0 / (1.0 - r2) if r2 < 0.999 else np.inf


def forward_select(Xdf, y, max_vif=MAX_VIF):
    cols = list(Xdf.columns)
    chosen = []
    best_adj_r2 = -np.inf
    while True:
        candidates = [c for c in cols if c not in chosen]
        trial_results = []
        for c in candidates:
            trial_cols = chosen + [c]
            Xtrial = Xdf[trial_cols].values
            # VIF check: the NEW column's VIF against the others already-plus-this-one
            if len(trial_cols) > 1:
                j = trial_cols.index(c)
                if _vif(Xtrial, j) > max_vif:
                    continue
            beta, fitted, resid = _ols(Xtrial, y)
            adj_r2 = _adj_r2(y, resid, len(trial_cols))
            trial_results.append((adj_r2, c))
        if not trial_results:
            break
        trial_results.sort(reverse=True)
        best_this_round, best_col = trial_results[0]
        if best_this_round <= best_adj_r2 + 1e-9:
            break
        chosen.append(best_col)
        best_adj_r2 = best_this_round
    return chosen, best_adj_r2


# =============================================================================
def fit_mlr(attr_df, lm_df, exclude_stations=None, max_vif=MAX_VIF):
    """attr_df: indexed by station, has the CAND_SPEC source columns.
    lm_df: indexed by station, has 'mean_ams'.
    exclude_stations: iterable of station names to drop from the TRAINING set (for
    leave-one-out use in step5.py) -- forward selection is rerun from scratch on what's
    left, so the chosen predictor set can legitimately differ run to run.
    Returns a dict model: predictors, beta, rmse_log, smearing, chosen columns' stats.
    """
    stations = attr_df.index.intersection(lm_df.index)
    if exclude_stations:
        stations = stations.difference(set(exclude_stations))
    Xall = build_candidates(attr_df.loc[stations])
    y = np.log10(lm_df.loc[stations, "mean_ams"].astype(float).clip(lower=1e-6))

    chosen, adj_r2 = forward_select(Xall, y, max_vif=max_vif)
    if not chosen:
        chosen = ["log10_area"]
    X = Xall[chosen].values
    beta, fitted, resid = _ols(X, y.values)
    rmse_log = float(np.sqrt(np.mean(resid ** 2)))
    smearing = float(np.mean(10 ** resid))  # Duan's smearing estimate

    return dict(predictors=chosen, beta=beta, adj_r2=float(_adj_r2(y.values, resid, len(chosen))),
                rmse_log=rmse_log, smearing=smearing, n_train=len(stations),
                train_stations=list(stations))


def predict_mu(model, attr_row):
    """attr_row: pandas Series with the CAND_SPEC source columns for ONE station."""
    x = np.array([1.0] + [build_candidates(attr_row.to_frame().T)[p].iloc[0]
                           for p in model["predictors"]])
    log10mu = float(x @ model["beta"])
    return model["smearing"] * (10 ** log10mu)


# =============================================================================
def main():
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)

    attr = pd.read_csv(os.path.join(IN1, "step1_final_attributes.csv")).set_index("station")
    lm = pd.read_csv(os.path.join(IN2, "step2_atsite_lmoments.csv")).set_index("station")

    model = fit_mlr(attr, lm)

    preds = []
    for st in model["train_stations"]:
        mu_hat = predict_mu(model, attr.loc[st])
        preds.append(dict(station=st, mu_at_site=lm.loc[st, "mean_ams"], mu_predicted=mu_hat,
                           pct_error=100 * (mu_hat - lm.loc[st, "mean_ams"]) / lm.loc[st, "mean_ams"]))
    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(f"{out}/step4_mu_predictions.csv", index=False)

    eqn = "log10(mu) = " + f"{model['beta'][0]:.4f}"
    for p, b in zip(model["predictors"], model["beta"][1:]):
        eqn += f" {'+' if b >= 0 else '-'} {abs(b):.4f}*{p}"
    with open(f"{out}/step4_indexflood_model.txt", "w") as fh:
        fh.write("STEP 4 - INDEX FLOOD (mu) REGRESSION\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Training sites: {model['n_train']}\n")
        fh.write(f"Predictors chosen (forward selection, VIF <= {MAX_VIF}): {model['predictors']}\n\n")
        fh.write(eqn + "\n")
        fh.write(f"\nmu_hat = smearing * 10**(fitted log10(mu))   [Duan bias correction]\n")
        fh.write(f"Duan smearing factor = {model['smearing']:.4f}\n")
        fh.write(f"Adjusted R^2 (log10 space) = {model['adj_r2']:.4f}\n")
        fh.write(f"Residual s.d. (log10) = {model['rmse_log']:.4f}\n")
        fh.write(f"In-sample mu %% error: median {pred_df.pct_error.abs().median():.1f}%%, "
                 f"IQR [{pred_df.pct_error.abs().quantile(.25):.1f}, "
                 f"{pred_df.pct_error.abs().quantile(.75):.1f}]%%\n")

    print("\n================ STEP 4 SUMMARY ================")
    print(f"Predictors: {model['predictors']}")
    print(eqn)
    print(f"Duan smearing factor: {model['smearing']:.4f}")
    print(f"Adjusted R^2 (log10 space): {model['adj_r2']:.4f}")
    print(f"Residual s.d. (log10): {model['rmse_log']:.4f}")
    print(f"In-sample |%% error| on mu: median {pred_df.pct_error.abs().median():.1f}%%")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
