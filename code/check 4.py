"""
DIAGNOSTIC — why is L-CV so heterogeneous?
Looks for the two things that usually cause it: zero-inflated (missing-as-zero)
records, and regulated catchments. Read-only, changes nothing.
"""
import os, glob, re
import numpy as np
import pandas as pd

BASE = r"C:\Users\bsnre\Downloads\chingka kalai"
STREAMFLOW_FOLDER = os.path.join(BASE, "streamflow data")   # <-- same path as Stage 1
MIN_N_OBS = 330

def norm_key(s):
    s = re.sub(r"\.(xlsx|xls|csv)$", "", str(s).strip(), flags=re.I)
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return re.sub(r"streamflow$", "", re.sub(r"^riverwaterdischarge", "", s))

def find_col(df, cands):
    lk = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for c in cands:
        k = re.sub(r"[^a-z0-9]", "", c.lower())
        if k in lk: return lk[k]
    return None

def read_any(p):
    return pd.read_csv(p) if p.lower().endswith(".csv") else pd.read_excel(p)

lm = pd.read_csv(os.path.join(BASE, "rfa_lmoments.csv"))
print(f"{len(lm)} contributing sites\n")
print("AT-SITE L-CV DISTRIBUTION")
print(lm["t"].describe().round(3).to_string())
h, edges = np.histogram(lm["t"].dropna(), bins=10)
for c, lo, hi in zip(h, edges[:-1], edges[1:]):
    print(f"  {lo:.2f}-{hi:.2f} | {'#'*int(c)} {c}")

recs = []
for f in sorted(glob.glob(os.path.join(STREAMFLOW_FOLDER, "*.xls*")) +
                glob.glob(os.path.join(STREAMFLOW_FOLDER, "*.csv"))):
    try:
        d = read_any(f); d.columns = [str(c).strip() for c in d.columns]
        c_q  = find_col(d, ["max_discharge_cumec","max_discharge","max_disch"])
        c_mn = find_col(d, ["min_discharge_cumec","min_discharge","min_disch"])
        c_av = find_col(d, ["mean_discharge_cumec","mean_discharge","mean_disch"])
        c_n  = find_col(d, ["n_obs","nobs"])
        if c_q is None: continue
        if c_n is not None:
            d = d[pd.to_numeric(d[c_n], errors="coerce") >= MIN_N_OBS]
        q = pd.to_numeric(d[c_q], errors="coerce").dropna()
        if len(q) < 15: continue
        rec = {"join_key": norm_key(os.path.basename(f)), "n": len(q),
               "qmax_max": q.max(), "qmax_min": q.min(),
               "ratio_max_min": q.max()/q.min() if q.min() > 0 else np.inf}
        if c_mn is not None:
            mn = pd.to_numeric(d[c_mn], errors="coerce")
            rec["frac_zero_minQ"] = float((mn <= 0).mean())
        if c_av is not None:
            av = pd.to_numeric(d[c_av], errors="coerce")
            rec["mean_of_meanQ"] = av.mean()
            rec["peak_to_mean"] = q.mean()/av.mean() if av.mean() > 0 else np.nan
        recs.append(rec)
    except Exception as e:
        print("  ! ", os.path.basename(f), e)

fl = pd.DataFrame(recs)
if len(fl) == 0:
    import sys
    print(f"No streamflow files read from: {STREAMFLOW_FOLDER}")
    print(f"  Folder exists: {os.path.isdir(STREAMFLOW_FOLDER)}")
    if os.path.isdir(STREAMFLOW_FOLDER):
        print(f"  Contents: {os.listdir(STREAMFLOW_FOLDER)[:5]}")
    sys.exit(1)

m = lm.merge(fl, on="join_key", how="inner")
print(f"\nMerged diagnostics for {len(m)} sites")

print("\nCORRELATION OF L-CV WITH DATA-QUALITY INDICATORS")
for c in ["frac_zero_minQ", "ratio_max_min", "peak_to_mean", "n_used", "mean_of_meanQ"]:
    if c in m.columns:
        sub = m[["t", c]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) > 5:
            print(f"  t vs {c:18s} r = {sub['t'].corr(sub[c]):+.3f}  (n={len(sub)})")

print("\n10 HIGHEST L-CV SITES")
cols = [c for c in ["join_key","t","t3","n_used","frac_zero_minQ",
                    "ratio_max_min","peak_to_mean"] if c in m.columns]
print(m.nlargest(10, "t")[cols].to_string(index=False))
print("\n10 LOWEST L-CV SITES")
print(m.nsmallest(10, "t")[cols].to_string(index=False))

if "frac_zero_minQ" in m.columns:
    hi = m[m["frac_zero_minQ"] > 0.5]["t"]
    lo = m[m["frac_zero_minQ"] <= 0.5]["t"]
    if len(hi) > 2 and len(lo) > 2:
        print(f"\nSITES WITH >50% ZERO MIN-FLOW YEARS: n={len(hi)}, mean L-CV={hi.mean():.3f}")
        print(f"OTHER SITES:                        n={len(lo)}, mean L-CV={lo.mean():.3f}")
        print("  A large gap here means zero-filled records are driving the heterogeneity.")