"""
RFA STAGE 5 — Index-flood regression + final ungauged quantile estimates
=========================================================================

The last piece. Stages 1-4 give the regional growth curve q(T). This gives the
index flood mu for a site with NO streamflow record, so that

    Q(T) = mu_hat * q(T)

can be applied anywhere in the study area.

METHOD
  Fit  log10(mu) = b0 + b1*log10(area) + b2*X2 + ...  on contributing sites.
  Log-log is standard: flood magnitude scales as a power of area, so the
  relationship is linear in logs and the errors are closer to constant variance.

  Predictors are chosen by forward selection on adjusted R^2, with a variance
  inflation check so collinear attributes don't both enter. Validation is
  leave-one-out, reported in PERCENT error on the original scale, because an
  R^2 of 0.9 in log space can still be a 50% error in cumecs.

  The log-space fit is bias-corrected on back-transformation (smearing estimate),
  otherwise mu is systematically under-predicted.

OUTPUT
  rfa_indexflood_model.txt   coefficients, diagnostics, the usable equation
  rfa_final_quantiles.csv    Q(T) for all target sites from attributes alone
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\bsnre\Downloads\chingka kalai"

IN_REGIONS   = os.path.join(BASE, "rfa_regions.csv")
IN_QUANT     = os.path.join(BASE, "rfa_quantiles.csv")
OUT_MODEL    = os.path.join(BASE, "rfa_indexflood_model.txt")
OUT_FINAL    = os.path.join(BASE, "rfa_final_quantiles.csv")

RETURN_PER = [2, 5, 10, 25, 50, 100]
MAX_VIF    = 5.0
EXCLUDE_SEASONAL = True
LCV_CAP    = None    # e.g. 0.65 to drop suspect high-L-CV sites; None = keep all


# =============================================================================
# OLS
# =============================================================================
def ols(X, y):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    n, p = X.shape
    dof = n - p
    sse = float(resid @ resid)
    sst = float(((y - y.mean())**2).sum())
    r2 = 1 - sse/sst
    adj = 1 - (1-r2)*(n-1)/dof if dof > 0 else np.nan
    se = np.sqrt(np.diag(np.linalg.inv(XtX))*sse/dof) if dof > 0 else np.full(p, np.nan)
    return beta, r2, adj, se, resid, np.sqrt(sse/dof) if dof > 0 else np.nan


def vif(X, j):
    """Variance inflation factor for column j (X includes an intercept at col 0)."""
    others = [c for c in range(X.shape[1]) if c != j and c != 0]
    if not others:
        return 1.0
    A = np.column_stack([np.ones(len(X))] + [X[:, c] for c in others])
    try:
        b = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.inf
    r = X[:, j] - A @ b
    sst = ((X[:, j]-X[:, j].mean())**2).sum()
    if sst <= 0:
        return np.inf
    r2 = 1 - (r @ r)/sst
    return np.inf if r2 >= 0.9999 else 1/(1-r2)


# =============================================================================
# LOAD
# =============================================================================
print("="*70); print("STAGE 5 — INDEX FLOOD REGRESSION"); print("="*70)

df = pd.read_csv(IN_REGIONS)
if EXCLUDE_SEASONAL:
    df.loc[df["station"].str.contains("seasonal", case=False, na=False),
           "is_contributing"] = False

fit_df = df[(df["is_contributing"] == True) & df["l1"].notna() & (df["l1"] > 0)].copy()
if LCV_CAP is not None:
    before = len(fit_df)
    fit_df = fit_df[fit_df["t"] <= LCV_CAP]
    print(f"L-CV cap {LCV_CAP}: dropped {before-len(fit_df)} sites")

print(f"Fitting on {len(fit_df)} gauged sites")

# candidate predictors, all obtainable at an ungauged site
CAND = {}
if "area_km2" in df.columns:
    CAND["log10_area"] = np.log10(df["area_km2"])
for c in ["mean_elev_m", "min_elev_m", "mean_slope_deg", "drain_density",
          "temp_max_C", "temp_min_C", "temp_mean_max_C", "temp_mean_min_C",
          "lat", "lon"]:
    if c in df.columns:
        CAND[c] = df[c]
for c in ["relief_m", "rain_annual_mean_mm"]:
    if c in df.columns and (df[c] > 0).all():
        CAND[f"log10_{c}"] = np.log10(df[c])

CANDS = pd.DataFrame(CAND)
y_all = np.log10(df["l1"].where(df["l1"] > 0))
mask = fit_df.index
Xc = CANDS.loc[mask]
y = y_all.loc[mask].to_numpy(float)
good = np.isfinite(y) & np.isfinite(Xc.to_numpy(float)).all(axis=1)
Xc, y = Xc[good], y[good]
names = list(Xc.columns)
print(f"Candidate predictors ({len(names)}): {names}\n")


# =============================================================================
# FORWARD SELECTION
# =============================================================================
chosen = []
best_adj = -np.inf
print("Forward selection (adjusted R^2, VIF-limited):")
while True:
    trial = None
    for c in names:
        if c in chosen:
            continue
        cols = chosen + [c]
        X = np.column_stack([np.ones(len(Xc))] + [Xc[k].to_numpy(float) for k in cols])
        if any(vif(X, j) > MAX_VIF for j in range(1, X.shape[1])):
            continue
        _, _, adj, _, _, _ = ols(X, y)
        if adj > best_adj + 1e-4 and (trial is None or adj > trial[1]):
            trial = (c, adj)
    if trial is None:
        break
    chosen.append(trial[0]); best_adj = trial[1]
    print(f"  + {trial[0]:24s} adj R^2 = {best_adj:.4f}")

if not chosen:
    chosen = ["log10_area"]
    print("  ! nothing met the criteria; falling back to log10_area alone")

X = np.column_stack([np.ones(len(Xc))] + [Xc[k].to_numpy(float) for k in chosen])
beta, r2, adj, se, resid, rmse_log = ols(X, y)

print(f"\nFINAL MODEL   R^2 = {r2:.4f}   adj R^2 = {adj:.4f}   "
      f"residual s.d. (log10) = {rmse_log:.4f}")
print(f"{'term':>26} {'coef':>10} {'std err':>10} {'t':>8}")
print(f"{'intercept':>26} {beta[0]:>10.4f} {se[0]:>10.4f} {beta[0]/se[0]:>8.2f}")
for i, c in enumerate(chosen, start=1):
    print(f"{c:>26} {beta[i]:>10.4f} {se[i]:>10.4f} {beta[i]/se[i]:>8.2f}")

# Duan smearing factor corrects the back-transform bias
smear = float(np.mean(10**resid))
print(f"\nSmearing correction factor: {smear:.4f}")
if "log10_area" in chosen:
    b = beta[chosen.index("log10_area")+1]
    print(f"Area exponent = {b:.3f}  "
          f"(typical published range 0.5-0.9 for index flood vs area)")


# =============================================================================
# LEAVE-ONE-OUT
# =============================================================================
n = len(y)
loo = np.empty(n)
for i in range(n):
    k = np.ones(n, bool); k[i] = False
    b, *_ = ols(X[k], y[k])
    loo[i] = X[i] @ b
obs, pred = 10**y, 10**loo * smear
pe = 100*(pred/obs - 1)
print(f"\nLEAVE-ONE-OUT (original scale, cumecs)")
print(f"  median abs error {np.median(np.abs(pe)):.1f}%   "
      f"mean bias {pe.mean():+.1f}%   RMSE {np.sqrt((pe**2).mean()):.1f}%")
print(f"  within +/-25%: {(np.abs(pe)<=25).mean()*100:.0f}%   "
      f"within +/-50%: {(np.abs(pe)<=50).mean()*100:.0f}%")
r2_loo = 1 - ((y-loo)**2).sum()/((y-y.mean())**2).sum()
print(f"  LOO R^2 (log space) = {r2_loo:.4f}")


# =============================================================================
# APPLY TO ALL TARGET SITES
# =============================================================================
Xall = np.column_stack([np.ones(len(CANDS))] +
                       [CANDS[k].to_numpy(float) for k in chosen])
ok = np.isfinite(Xall).all(axis=1)
mu_hat = np.full(len(CANDS), np.nan)
mu_hat[ok] = 10**(Xall[ok] @ beta) * smear
df["index_flood_pred"] = mu_hat
df["index_flood_obs"] = df["l1"]

q = pd.read_csv(IN_QUANT)
roi = q[q["method"] == "ROI"].drop_duplicates(subset="station").set_index("station")
if q[q["method"] == "ROI"]["station"].duplicated().any():
    print("! duplicate station names in quantiles; kept the first of each for mapping")
gcols = [f"growth_T{T}" for T in RETURN_PER if f"growth_T{T}" in roi.columns]

out = df[["station", "region", "area_km2", "is_contributing",
          "index_flood_obs", "index_flood_pred"]].copy()
for T in RETURN_PER:
    gc = f"growth_T{T}"
    if gc in roi.columns:
        g = out["station"].map(roi[gc])
        out[f"growth_T{T}"] = g
        out[f"Q{T}_ungauged"] = out["index_flood_pred"] * g
        out[f"Q{T}_gauged"] = out["index_flood_obs"] * g

out.to_csv(OUT_FINAL, index=False)

with open(OUT_MODEL, "w") as fh:
    fh.write("INDEX FLOOD MODEL\n" + "="*60 + "\n\n")
    fh.write(f"Fitted on {len(y)} gauged sites\n")
    fh.write(f"R2={r2:.4f}  adjR2={adj:.4f}  LOO R2={r2_loo:.4f}\n")
    fh.write(f"Residual s.d. (log10) = {rmse_log:.4f}\n")
    fh.write(f"Smearing factor = {smear:.4f}\n\n")
    fh.write("log10(index_flood) = "
             + f"{beta[0]:.4f}"
             + "".join(f" {b:+.4f}*{c}" for b, c in zip(beta[1:], chosen)) + "\n")
    fh.write(f"index_flood = {smear:.4f} * 10^(above)\n\n")
    fh.write(f"LOO median abs error {np.median(np.abs(pe)):.1f}%, "
             f"RMSE {np.sqrt((pe**2).mean()):.1f}%\n")

print(f"\nUSABLE EQUATION")
print("  log10(mu) = " + f"{beta[0]:.4f}" +
      "".join(f" {b:+.4f}*{c}" for b, c in zip(beta[1:], chosen)))
print(f"  mu = {smear:.4f} * 10^(above)      then  Q(T) = mu * q(T)")
print(f"\nPredicted index flood for {int(np.isfinite(mu_hat).sum())}/{len(df)} sites")
print(f"Saved: {OUT_MODEL}\nSaved: {OUT_FINAL}")