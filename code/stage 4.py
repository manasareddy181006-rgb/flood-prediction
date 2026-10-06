"""
RFA STAGE 4 — Region of Influence, growth curves, and ROI-vs-clustering comparison
===================================================================================

Input : rfa_regions.csv  (PC scores, cluster labels, at-site L-moments)
Output: rfa_roi_pools.csv      each target site's pooling group + its H1
        rfa_quantiles.csv      Q(T) for all 276 target sites, both methods
        rfa_validation.csv     leave-one-out errors, ROI vs fixed clusters

ROI (Burn 1990; Zrinji & Burn 1994)
  There are no fixed regions. For each TARGET site, contributing sites are ranked
  by Euclidean distance in PC space and added one at a time until the pool holds
  5*T station-years (the "5T rule"). The pool is then trimmed while H1 stays high,
  so each site gets its own bespoke, homogeneous pooling group.

INDEX FLOOD
  Q(T) at site i = mu_i * q(T), where mu_i is the at-site mean annual max (the
  index flood) and q(T) is the dimensionless regional growth curve fitted to the
  pooled, at-site-mean-scaled L-moments.

  For a genuinely ungauged site mu_i must come from a regression on catchment
  attributes. That regression is fitted and reported here as well, because it is
  the step that makes the whole exercise usable.

VALIDATION
  Each contributing site is treated as ungauged in turn: it is removed from its
  own pool, Q(T) is estimated from the remaining sites, and the result is compared
  with its at-site estimate. Done for ROI and for the fixed clusters, so the two
  approaches are compared on identical sites.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gammaln
from scipy.optimize import fsolve

# =============================================================================
# CONFIG
# =============================================================================
BASE = r"C:\Users\bsnre\Downloads\chingka kalai"

IN_REGIONS    = os.path.join(BASE, "rfa_regions.csv")
OUT_POOLS     = os.path.join(BASE, "rfa_roi_pools.csv")
OUT_QUANTILES = os.path.join(BASE, "rfa_quantiles.csv")
OUT_VALID     = os.path.join(BASE, "rfa_validation.csv")
OUT_REPORT    = os.path.join(BASE, "rfa_stage4_report.txt")

TARGET_T   = 100          # design return period for the 5T rule
RETURN_PER = [2, 5, 10, 25, 50, 100]
MIN_POOL   = 5            # minimum sites in a pool
MAX_POOL   = 25           # cap; ROI pools should stay tight
H1_TARGET  = 2.0          # trim the pool while H1 exceeds this
NSIM_H     = 200          # replicates for the pool H1 test (500 for final run)
EXCLUDE_SEASONAL = True   # drop "(seasonal)" gauges from the contributing set
SEED = 42

# =============================================================================
# L-MOMENTS / DISTRIBUTIONS
# =============================================================================
def samlmu(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    n = x.size
    if n < 4: return np.array([np.nan]*5)
    j = np.arange(1, n+1); b = np.zeros(4); b[0] = x.mean()
    for r in range(1, 4):
        num, den = np.ones(n), 1.0
        for k in range(r): num = num*(j-1-k); den = den*(n-1-k)
        b[r] = np.sum(num*x)/den/n
    l1 = b[0]; l2 = 2*b[1]-b[0]
    l3 = 6*b[2]-6*b[1]+b[0]; l4 = 20*b[3]-30*b[2]+12*b[1]-b[0]
    return np.array([l1, l2, l2/l1, l3/l2, l4/l2])

# --- GNO (generalised normal / 3-parameter lognormal) ---
_A = [2.0466534, -3.6544371, 1.8396733, -0.20360244]
_B = [1.0, -2.0182173, 1.2033464, -0.40077113]

def gno_fit(l1, l2, t3):
    t2 = t3*t3
    k = -t3*(_A[0]+_A[1]*t2+_A[2]*t2**2+_A[3]*t2**3) / \
        (_B[0]+_B[1]*t2+_B[2]*t2**2+_B[3]*t2**3)
    if abs(k) < 1e-8:
        return 0.0, l2*np.sqrt(np.pi), l1
    a = l2*k*np.exp(-k*k/2)/(1-2*norm.cdf(-k/np.sqrt(2)))
    return k, a, l1 - a*(1-np.exp(k*k/2))/k

def gno_q(F, k, a, xi):
    z = norm.ppf(F)
    return xi + (a*(1-np.exp(-k*z))/k if abs(k) > 1e-8 else a*z)

# --- GEV ---
def gev_fit(l1, l2, t3):
    c = 2/(3+t3) - np.log(2)/np.log(3)
    k = 7.8590*c + 2.9554*c*c
    if abs(k) < 1e-8:
        a = l2/np.log(2); return 0.0, a, l1 - a*0.5772156649
    g = np.exp(gammaln(1+k))
    a = l2*k/((1-2**-k)*g)
    return k, a, l1 - a*(1-g)/k

def gev_q(F, k, a, xi):
    y = -np.log(F)
    return xi + (a*(1-y**k)/k if abs(k) > 1e-8 else -a*np.log(y))

FITTERS = {"GNO": (gno_fit, gno_q), "GEV": (gev_fit, gev_q)}

# --- kappa (for the H1 null) ---
def kappa_ratios(k, h):
    def g(r):
        if abs(h) < 1e-8: return np.exp(gammaln(1+k))*r**(-k)
        if h > 0:
            if 1+k+r/h <= 0 or r/h <= 0: return np.nan
            return r*np.exp(gammaln(1+k)+gammaln(r/h)-gammaln(1+k+r/h))/h**(1+k)
        if -k-r/h <= 0 or 1-r/h <= 0: return np.nan
        return r*np.exp(gammaln(1+k)+gammaln(-k-r/h)-gammaln(1-r/h))/(-h)**(1+k)
    g1, g2, g3, g4 = g(1), g(2), g(3), g(4)
    if not np.all(np.isfinite([g1,g2,g3,g4])) or abs(g1-g2) < 1e-13:
        return np.nan, np.nan
    return ((-g1+3*g2-2*g3)/(g1-g2), (g1-6*g2+10*g3-5*g4)/(g1-g2))

def fit_kappa(t3, t4):
    def eqs(p):
        a, b = kappa_ratios(p[0], p[1])
        return [1e3, 1e3] if not np.isfinite(a) or not np.isfinite(b) else [a-t3, b-t4]
    for guess in [(-0.1,0.0),(0.0,0.0),(0.1,-0.2),(-0.2,0.3),(0.05,0.5)]:
        try:
            sol, _, ier, _ = fsolve(eqs, guess, full_output=True)
            if ier == 1:
                a, b = kappa_ratios(sol[0], sol[1])
                if np.isfinite(a) and abs(a-t3) < 1e-4 and abs(b-t4) < 1e-4 \
                   and sol[0] > -1 and sol[1] > -1:
                    return sol[0], sol[1]
        except Exception: pass
    return None

def kappa_sample(n, k, h, rs):
    F = rs.uniform(1e-10, 1-1e-10, n)
    inner = -np.log(F) if abs(h) < 1e-8 else (1-F**h)/h
    return 1.0 - np.log(inner) if abs(k) < 1e-8 else 1.0 + (1-inner**k)/k

def H1(t, t3, t4, n, nsim=NSIM_H, rs=None):
    rs = rs or np.random.default_rng(SEED)
    w = n/np.sum(n)
    tR, t3R, t4R = np.sum(w*t), np.sum(w*t3), np.sum(w*t4)
    V = np.sqrt(np.sum(n*(t-tR)**2)/np.sum(n))
    fit = fit_kappa(t3R, t4R)
    if fit is None: return np.nan, tR, t3R, t4R
    k, h = fit
    base = samlmu(kappa_sample(100000, k, h, np.random.default_rng(999)))
    shift = base[1]/tR - base[0]
    sims = []
    for _ in range(nsim):
        tt = []
        for ni in n:
            L = samlmu(kappa_sample(int(ni), k, h, rs) + shift)
            if np.isfinite(L[2]): tt.append(L[2])
        tt = np.asarray(tt)
        if len(tt) > 1:
            ww = n[:len(tt)]/np.sum(n[:len(tt)])
            sims.append(np.sqrt(np.sum(n[:len(tt)]*(tt-np.sum(ww*tt))**2)/np.sum(n[:len(tt)])))
    sims = np.asarray(sims)
    if len(sims) < 10 or sims.std() == 0: return np.nan, tR, t3R, t4R
    return (V-sims.mean())/sims.std(ddof=1), tR, t3R, t4R


# =============================================================================
# LOAD
# =============================================================================
print("="*70); print("STAGE 4 — ROI + GROWTH CURVES"); print("="*70)

df = pd.read_csv(IN_REGIONS)
if EXCLUDE_SEASONAL:
    seas = df["station"].str.contains("seasonal", case=False, na=False)
    df.loc[seas, "is_contributing"] = False
    print(f"Excluded {int(seas.sum())} seasonal gauges from the contributing set")

PCS = [c for c in df.columns if c.startswith("PC")]
con = df[(df["is_contributing"] == True) &
         df[["t","t3","t4","n_used","l1"]].notna().all(axis=1)].reset_index(drop=True)
print(f"Target sites: {len(df)}   Contributing: {len(con)}   PC dims: {len(PCS)}")
print(f"5T rule: pools need {5*TARGET_T} station-years for T={TARGET_T}\n")

C = con[PCS].to_numpy(float)
cn = con["n_used"].to_numpy(float)
ct, ct3, ct4 = (con[c].to_numpy(float) for c in ["t","t3","t4"])


# =============================================================================
# ROI POOLING
# =============================================================================
def build_pool(pc, exclude_idx=None, target_years=5*TARGET_T):
    """Nearest contributing sites in PC space until the 5T rule is met."""
    d = np.linalg.norm(C - pc, axis=1)
    if exclude_idx is not None: d[exclude_idx] = np.inf
    order = np.argsort(d)
    order = order[np.isfinite(d[order])]
    pool, yrs = [], 0
    for i in order:
        pool.append(i); yrs += cn[i]
        if yrs >= target_years and len(pool) >= MIN_POOL: break
        if len(pool) >= MAX_POOL: break
    return np.array(pool)

def trim_pool(pool, rs):
    """Drop the most distant site while H1 exceeds target and pool stays legal."""
    h, *_ = H1(ct[pool], ct3[pool], ct4[pool], cn[pool], rs=rs)
    while np.isfinite(h) and h > H1_TARGET and len(pool) > MIN_POOL:
        cand = pool[:-1]
        if cn[cand].sum() < 5*TARGET_T*0.6: break
        h2, *_ = H1(ct[cand], ct3[cand], ct4[cand], cn[cand], rs=rs)
        if not np.isfinite(h2) or h2 >= h: break
        pool, h = cand, h2
    return pool, h

print("Building ROI pools (H1-trimmed)...")
rs = np.random.default_rng(SEED)
pool_rows = []
for i, r in df.iterrows():
    pc = r[PCS].to_numpy(float)
    if not np.all(np.isfinite(pc)): continue
    self_idx = None
    hit = con.index[con["station"] == r["station"]]
    if len(hit): self_idx = hit[0]
    pool = build_pool(pc)
    pool, h = trim_pool(pool, rs)
    pool_rows.append({"station": r["station"], "region": r["region"],
                      "pool_size": len(pool), "pool_years": int(cn[pool].sum()),
                      "H1": h, "is_contributing": bool(r["is_contributing"]),
                      "pool": ";".join(con.loc[pool, "station"].astype(str))})
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(df)}")

pools = pd.DataFrame(pool_rows)
pools.to_csv(OUT_POOLS, index=False)
hh = pools["H1"].dropna()
print(f"\nROI pool H1:  median={hh.median():.2f}  "
      f"<1: {(hh<1).sum()}  1-2: {((hh>=1)&(hh<2)).sum()}  >=2: {(hh>=2).sum()}  "
      f"of {len(hh)}")
print(f"Pool size: median={pools['pool_size'].median():.0f}  "
      f"years: median={pools['pool_years'].median():.0f}")


# =============================================================================
# GROWTH CURVES + QUANTILES
# =============================================================================
def growth_curve(pool, dist="GNO"):
    """Record-length-weighted regional L-moments of at-site-mean-scaled data."""
    w = cn[pool]/cn[pool].sum()
    tR, t3R = np.sum(w*ct[pool]), np.sum(w*ct3[pool])
    fit, q = FITTERS[dist]
    k, a, xi = fit(1.0, tR, t3R)      # l1 = 1 by construction after scaling
    return [q(1-1/T, k, a, xi) for T in RETURN_PER], (k, a, xi), (tR, t3R)

print("\nEstimating quantiles...")
qrows = []
for _, pr in pools.iterrows():
    names = pr["pool"].split(";")
    pool = con.index[con["station"].astype(str).isin(names)].to_numpy()
    if len(pool) < MIN_POOL: continue
    gc, _, (tR, t3R) = growth_curve(pool)
    src = df[df["station"] == pr["station"]]
    mu = float(src["l1"].iloc[0]) if src["l1"].notna().any() else np.nan
    row = {"station": pr["station"], "region": pr["region"], "method": "ROI",
           "index_flood": mu, "pool_size": pr["pool_size"], "H1": pr["H1"],
           "regional_LCV": tR, "regional_Lskew": t3R}
    for T, g in zip(RETURN_PER, gc):
        row[f"growth_T{T}"] = g
        row[f"Q{T}"] = mu*g if np.isfinite(mu) else np.nan
    qrows.append(row)

# fixed-cluster comparison
for reg in sorted(con["region"].unique()):
    pool = con.index[con["region"] == reg].to_numpy()
    if len(pool) < MIN_POOL: continue
    gc, _, (tR, t3R) = growth_curve(pool)
    for _, src in df[df["region"] == reg].iterrows():
        mu = src["l1"] if np.isfinite(src["l1"]) else np.nan
        row = {"station": src["station"], "region": reg, "method": "CLUSTER",
               "index_flood": mu, "pool_size": len(pool), "H1": np.nan,
               "regional_LCV": tR, "regional_Lskew": t3R}
        for T, g in zip(RETURN_PER, gc):
            row[f"growth_T{T}"] = g
            row[f"Q{T}"] = mu*g if np.isfinite(mu) else np.nan
        qrows.append(row)

quant = pd.DataFrame(qrows)
quant.to_csv(OUT_QUANTILES, index=False)
print(f"Quantiles for {quant['station'].nunique()} sites x 2 methods")


# =============================================================================
# LEAVE-ONE-OUT VALIDATION
# =============================================================================
print("\nLeave-one-out validation...")
vrows = []
for i in range(len(con)):
    at = con.iloc[i]
    k_a, a_a, xi_a = gno_fit(1.0, at["t"], at["t3"])
    at_q = {T: gno_q(1-1/T, k_a, a_a, xi_a) for T in RETURN_PER}

    pool = build_pool(C[i], exclude_idx=i)
    pool = pool[pool != i]
    roi_q = {}
    if len(pool) >= MIN_POOL:
        gc, _, _ = growth_curve(pool)
        roi_q = dict(zip(RETURN_PER, gc))

    cl = np.array([j for j in range(len(con))
                   if con.iloc[j]["region"] == at["region"] and j != i])
    cl_q = {}
    if len(cl) >= MIN_POOL:
        gc, _, _ = growth_curve(cl)
        cl_q = dict(zip(RETURN_PER, gc))

    for T in RETURN_PER:
        vrows.append({"station": at["station"], "T": T, "at_site": at_q[T],
                      "roi": roi_q.get(T, np.nan), "cluster": cl_q.get(T, np.nan)})

val = pd.DataFrame(vrows)
val["roi_err_pct"] = 100*(val["roi"]/val["at_site"] - 1)
val["cluster_err_pct"] = 100*(val["cluster"]/val["at_site"] - 1)
val.to_csv(OUT_VALID, index=False)

print("\n" + "="*70); print("ROI vs FIXED CLUSTERS — leave-one-out"); print("="*70)
print(f"{'T':>5} {'ROI bias%':>10} {'ROI RMSE%':>10} {'CLU bias%':>10} "
      f"{'CLU RMSE%':>10} {'winner':>8}")
lines = []
for T in RETURN_PER:
    s = val[val["T"] == T]
    rb, re_ = s["roi_err_pct"].mean(), np.sqrt((s["roi_err_pct"]**2).mean())
    cb, ce = s["cluster_err_pct"].mean(), np.sqrt((s["cluster_err_pct"]**2).mean())
    win = "ROI" if re_ < ce else "cluster"
    line = f"{T:>5} {rb:>10.1f} {re_:>10.1f} {cb:>10.1f} {ce:>10.1f} {win:>8}"
    print(line); lines.append(line)

with open(OUT_REPORT, "w") as fh:
    fh.write("RFA STAGE 4 — ROI vs CLUSTERING\n" + "="*60 + "\n\n")
    fh.write(f"Contributing sites: {len(con)}   Targets: {len(df)}\n")
    fh.write(f"ROI pool H1 median: {hh.median():.2f}\n")
    fh.write(f"  H1<1: {(hh<1).sum()}   1-2: {((hh>=1)&(hh<2)).sum()}   "
             f">=2: {(hh>=2).sum()}\n\n")
    fh.write(f"{'T':>5} {'ROI bias%':>10} {'ROI RMSE%':>10} {'CLU bias%':>10} "
             f"{'CLU RMSE%':>10} {'winner':>8}\n")
    fh.write("\n".join(lines) + "\n")

print(f"\nSaved: {OUT_POOLS}\nSaved: {OUT_QUANTILES}")
print(f"Saved: {OUT_VALID}\nSaved: {OUT_REPORT}")
print("\nNOTE: Q(T) here uses each site's OWN mean annual max as the index flood.")
print("For truly ungauged sites, regress log10(index_flood) on log10(area) and the")
print("other attributes — that regression is the last piece needed for application.")