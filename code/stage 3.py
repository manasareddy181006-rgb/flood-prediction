"""
RFA STAGE 3 — Within-region discordancy, heterogeneity (H1/H2/H3), goodness of fit
===================================================================================

Input : rfa_regions.csv, rfa_lmoments.csv   (Stages 1-2)
Output: rfa_homogeneity.csv, rfa_stage3_report.txt

METHOD — Hosking & Wallis (1997)
  Di   within each region on (t, t3, t4); Di > crit(N) => site may not belong.
  H1   based on weighted s.d. of at-site L-CV        <- the primary test
  H2   based on L-CV / L-skew distance
  H3   based on L-skew / L-kurtosis distance
       H < 1     acceptably homogeneous
       1 <= H < 2  possibly heterogeneous
       H >= 2    definitely heterogeneous -> adjust the region
  Z^DIST  goodness of fit for GLO / GEV / GNO / PE3 / GPA; |Z| <= 1.64 accepted.

The null distribution for H comes from NSIM simulated homogeneous regions drawn
from a 4-parameter kappa fitted to the regional average L-moments, with each
simulated site given the same record length as the real site it stands for.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import fsolve
from scipy.stats import norm

# =============================================================================
# CONFIG
# =============================================================================
BASE = r"C:\Users\bsnre\Downloads\chingka kalai"

IN_REGIONS  = os.path.join(BASE, "rfa_regions.csv")
OUT_CSV     = os.path.join(BASE, "rfa_homogeneity.csv")
OUT_REPORT  = os.path.join(BASE, "rfa_stage3_report.txt")

NSIM = 500          # H&W recommend 500. Raise to 1000 for the final run.
SEED = 42
rng = np.random.default_rng(SEED)

# Hosking & Wallis discordancy critical values by number of sites in region
DCRIT = {5: 1.333, 6: 1.648, 7: 1.917, 8: 2.140, 9: 2.329,
         10: 2.491, 11: 2.632, 12: 2.757, 13: 2.869, 14: 2.971}
def dcrit(n):
    return DCRIT.get(n, 3.0) if n < 15 else 3.0


# =============================================================================
# L-MOMENTS
# =============================================================================
def samlmu(x):
    x = np.sort(np.asarray(x, float)); x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return np.array([np.nan]*5)
    j = np.arange(1, n+1)
    b = np.zeros(4); b[0] = x.mean()
    for r in range(1, 4):
        num, den = np.ones(n), 1.0
        for k in range(r):
            num = num*(j-1-k); den = den*(n-1-k)
        b[r] = np.sum(num*x)/den/n
    l1 = b[0]; l2 = 2*b[1]-b[0]
    l3 = 6*b[2]-6*b[1]+b[0]; l4 = 20*b[3]-30*b[2]+12*b[1]-b[0]
    return np.array([l1, l2, l2/l1, l3/l2, l4/l2])


def discordancy(U):
    U = np.asarray(U, float); N = U.shape[0]
    if N < 4:
        return np.full(N, np.nan)
    D = U - U.mean(axis=0)
    S = D.T @ D
    try:
        Si = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Si = np.linalg.pinv(S)
    return np.array([(N/3.0)*(d @ Si @ d) for d in D])


# =============================================================================
# KAPPA DISTRIBUTION  (h=-1 GLO, h=0 GEV, h=+1 GPA)
# =============================================================================
def kappa_ratios(k, h):
    def g(r):
        if abs(h) < 1e-8:
            return np.exp(gammaln(1+k))*r**(-k)
        if h > 0:
            if 1+k+r/h <= 0 or r/h <= 0: return np.nan
            return r*np.exp(gammaln(1+k)+gammaln(r/h)-gammaln(1+k+r/h))/h**(1+k)
        if -k-r/h <= 0 or 1-r/h <= 0: return np.nan
        return r*np.exp(gammaln(1+k)+gammaln(-k-r/h)-gammaln(1-r/h))/(-h)**(1+k)
    g1, g2, g3, g4 = g(1), g(2), g(3), g(4)
    if not np.all(np.isfinite([g1, g2, g3, g4])) or abs(g1-g2) < 1e-13:
        return np.nan, np.nan
    return ((-g1+3*g2-2*g3)/(g1-g2), (g1-6*g2+10*g3-5*g4)/(g1-g2))


def fit_kappa(t3, t4):
    """Solve (k,h) matching target t3,t4. Returns (k,h) or None."""
    def eqs(p):
        a, b = kappa_ratios(p[0], p[1])
        if not np.isfinite(a) or not np.isfinite(b):
            return [1e3, 1e3]
        return [a-t3, b-t4]
    for guess in [(-0.1, 0.0), (0.0, 0.0), (0.1, -0.2), (-0.2, 0.3),
                  (0.05, 0.5), (-0.3, -0.5), (0.2, 0.2)]:
        try:
            sol, _, ier, _ = fsolve(eqs, guess, full_output=True)
            if ier == 1:
                a, b = kappa_ratios(sol[0], sol[1])
                if np.isfinite(a) and abs(a-t3) < 1e-4 and abs(b-t4) < 1e-4 \
                   and sol[0] > -1 and sol[1] > -1:
                    return sol[0], sol[1]
        except Exception:
            pass
    return None


def kappa_quantile(F, k, h, xi=1.0, alpha=1.0):
    F = np.asarray(F, float)
    if abs(h) < 1e-8:
        inner = -np.log(F)
    else:
        inner = (1-F**h)/h
    if abs(k) < 1e-8:
        return xi - alpha*np.log(inner)
    return xi + (alpha/k)*(1-inner**k)


def kappa_sample(n, k, h, rs):
    return kappa_quantile(rs.uniform(1e-10, 1-1e-10, n), k, h)


# =============================================================================
# HETEROGENEITY
# =============================================================================
def V_stats(t, t3, t4, n):
    w = n/np.sum(n)
    tR, t3R, t4R = np.sum(w*t), np.sum(w*t3), np.sum(w*t4)
    V1 = np.sqrt(np.sum(n*(t-tR)**2)/np.sum(n))
    V2 = np.sum(n*np.sqrt((t-tR)**2+(t3-t3R)**2))/np.sum(n)
    V3 = np.sum(n*np.sqrt((t3-t3R)**2+(t4-t4R)**2))/np.sum(n)
    return (V1, V2, V3), (tR, t3R, t4R)


def heterogeneity(t, t3, t4, n, nsim=NSIM, rs=None):
    rs = rs or np.random.default_rng(SEED)
    (V1o, V2o, V3o), (tR, t3R, t4R) = V_stats(t, t3, t4, n)
    fit = fit_kappa(t3R, t4R)
    if fit is None:
        return dict(H1=np.nan, H2=np.nan, H3=np.nan, kappa=None,
                    tR=tR, t3R=t3R, t4R=t4R, note="kappa fit failed")
    k, h = fit
    # scale kappa so its L-CV equals the regional average
    base = kappa_sample(200000, k, h, np.random.default_rng(999))
    lm = samlmu(base)
    shift = lm[1]/tR - lm[0]          # translate so that l2/l1 == tR

    sims = np.empty((nsim, 3))
    for s in range(nsim):
        tt = np.empty(len(n)); t3s = np.empty(len(n)); t4s = np.empty(len(n))
        for i, ni in enumerate(n):
            x = kappa_sample(int(ni), k, h, rs) + shift
            L = samlmu(x)
            tt[i], t3s[i], t4s[i] = L[2], L[3], L[4]
        ok = np.isfinite(tt) & np.isfinite(t3s) & np.isfinite(t4s)
        sims[s] = V_stats(tt[ok], t3s[ok], t4s[ok], np.asarray(n)[ok])[0] \
            if ok.sum() > 1 else (np.nan, np.nan, np.nan)

    out = dict(kappa=(k, h), tR=tR, t3R=t3R, t4R=t4R, note="")
    for i, (name, obs) in enumerate(zip(["H1", "H2", "H3"], [V1o, V2o, V3o])):
        col = sims[:, i]; col = col[np.isfinite(col)]
        out[name] = (obs-col.mean())/col.std(ddof=1) if len(col) > 10 and col.std() > 0 else np.nan
    return out


def verdict(H):
    if not np.isfinite(H):      return "n/a"
    if H < 1:                   return "HOMOGENEOUS"
    if H < 2:                   return "possibly heterog."
    return "HETEROGENEOUS"


# =============================================================================
# GOODNESS OF FIT  (Z^DIST)
# =============================================================================
def dist_t4(name, t3):
    """t4 as a function of t3 for each candidate 3-parameter distribution."""
    A = {"GLO": [0.16667, 0.83333, 0.0, 0.0, 0.0],
         "GEV": [0.10701, 0.11090, 0.84838, -0.06669, 0.00567],
         "GNO": [0.12282, 0.0, 0.77518, 0.0, 0.12279],
         "PE3": [0.12240, 0.0, 0.30115, 0.0, 0.95812],
         "GPA": [0.20196, 0.95924, -0.20096, 0.04061, 0.0]}
    c = A[name]
    if name == "GLO":
        return c[0] + c[1]*t3**2
    if name in ("GNO", "PE3"):
        return c[0] + c[2]*t3**2 + c[4]*t3**4
    return c[0] + c[1]*t3 + c[2]*t3**2 + c[3]*t3**3 + c[4]*t3**4


def goodness_of_fit(t, t3, t4, n, nsim=NSIM, rs=None):
    rs = rs or np.random.default_rng(SEED)
    w = n/np.sum(n)
    t3R, t4R = np.sum(w*t3), np.sum(w*t4)
    fit = fit_kappa(t3R, t4R)
    if fit is None:
        return {}, t3R, t4R
    k, h = fit
    sim_t4 = np.empty(nsim); sim_t3 = np.empty(nsim)
    for s in range(nsim):
        a = np.empty(len(n)); b = np.empty(len(n))
        for i, ni in enumerate(n):
            L = samlmu(kappa_sample(int(ni), k, h, rs))
            a[i], b[i] = L[3], L[4]
        ok = np.isfinite(a) & np.isfinite(b)
        ww = np.asarray(n)[ok]/np.sum(np.asarray(n)[ok])
        sim_t3[s], sim_t4[s] = np.sum(ww*a[ok]), np.sum(ww*b[ok])
    B4 = np.mean(sim_t4) - t4R
    S4 = np.std(sim_t4, ddof=1)
    res = {}
    for d in ["GLO", "GEV", "GNO", "PE3", "GPA"]:
        res[d] = (dist_t4(d, t3R) - t4R + B4)/S4 if S4 > 0 else np.nan
    return res, t3R, t4R


# =============================================================================
# RUN
# =============================================================================
print("="*70); print("STAGE 3 — REGIONAL TESTS"); print("="*70)

df = pd.read_csv(IN_REGIONS)
con = df[df["is_contributing"] == True].copy()
con = con[con[["t", "t3", "t4", "n_used"]].notna().all(axis=1)].reset_index(drop=True)
print(f"Contributing sites: {len(con)}   Regions: {sorted(con['region'].unique())}")
print(f"Monte Carlo simulations per region: {NSIM}\n")

rows, report = [], []
for reg in sorted(con["region"].unique()):
    g = con[con["region"] == reg].reset_index(drop=True)
    N = len(g)
    head = f"REGION {reg} — {N} contributing sites, {int(g['n_used'].sum())} station-years"
    print("="*70); print(head); print("="*70)
    report.append("\n" + "="*70 + f"\n{head}\n" + "="*70)

    if N < 5:
        msg = f"  Only {N} sites — too few to test. Merge this region."
        print(msg); report.append(msg)
        for _, r in g.iterrows():
            rows.append({"region": reg, "station": r["station"], "n_used": r["n_used"],
                         "t": r["t"], "t3": r["t3"], "t4": r["t4"],
                         "Di_region": np.nan, "discordant": False})
        continue

    # --- within-region discordancy ---
    U = g[["t", "t3", "t4"]].to_numpy(float)
    Di = discordancy(U)
    g["Di_region"] = Di
    crit = dcrit(N)
    g["discordant"] = Di > crit
    print(f"\nDiscordancy (critical value for N={N}: {crit:.3f})")
    report.append(f"\nDiscordancy critical value (N={N}): {crit:.3f}")
    bad = g[g["discordant"]].sort_values("Di_region", ascending=False)
    if len(bad):
        for _, r in bad.iterrows():
            line = (f"  DISCORDANT  {str(r['station'])[:32]:34s} Di={r['Di_region']:5.2f} "
                    f"n={int(r['n_used']):3d}  t={r['t']:.3f} t3={r['t3']:+.3f}")
            print(line); report.append(line)
    else:
        print("  No discordant sites."); report.append("  No discordant sites.")

    # --- heterogeneity, before and after removing discordant sites ---
    def run(sub, tag):
        H = heterogeneity(sub["t"].to_numpy(float), sub["t3"].to_numpy(float),
                          sub["t4"].to_numpy(float), sub["n_used"].to_numpy(float),
                          rs=np.random.default_rng(SEED))
        lines = [f"\n{tag}  ({len(sub)} sites)",
                 f"  Regional avg:  L-CV={H['tR']:.4f}  L-skew={H['t3R']:.4f}  "
                 f"L-kurt={H['t4R']:.4f}"]
        if H["kappa"]:
            lines.append(f"  Fitted kappa:  k={H['kappa'][0]:+.4f}  h={H['kappa'][1]:+.4f}")
        else:
            lines.append(f"  ! {H['note']}")
        for hn in ["H1", "H2", "H3"]:
            v = H[hn]
            lines.append(f"  {hn} = {v:6.2f}   {verdict(v)}" if np.isfinite(v)
                         else f"  {hn} =    n/a")
        for l in lines:
            print(l)
        report.extend(lines)
        return H

    H_all = run(g, "HETEROGENEITY — all sites")
    g_clean = g[~g["discordant"]].reset_index(drop=True)
    H_clean = run(g_clean, "HETEROGENEITY — discordant removed") if \
        (g["discordant"].any() and len(g_clean) >= 5) else H_all

    # --- goodness of fit on the cleaned region ---
    use = g_clean if len(g_clean) >= 5 else g
    Z, t3R, t4R = goodness_of_fit(use["t"].to_numpy(float), use["t3"].to_numpy(float),
                                  use["t4"].to_numpy(float), use["n_used"].to_numpy(float),
                                  rs=np.random.default_rng(SEED))
    if Z:
        lines = ["\nGOODNESS OF FIT (|Z| <= 1.64 accepted)"]
        for d, z in sorted(Z.items(), key=lambda kv: abs(kv[1])):
            lines.append(f"  {d}: Z = {z:+6.2f}   {'ACCEPT' if abs(z) <= 1.64 else 'reject'}")
        best = min(Z.items(), key=lambda kv: abs(kv[1]))
        lines.append(f"  -> best fit: {best[0]}")
        for l in lines: print(l)
        report.extend(lines)
    else:
        print("\nGOODNESS OF FIT: kappa fit failed, cannot compute Z.")

    for _, r in g.iterrows():
        rows.append({"region": reg, "station": r["station"], "n_used": r["n_used"],
                     "t": r["t"], "t3": r["t3"], "t4": r["t4"],
                     "Di_region": r["Di_region"], "discordant": bool(r["discordant"]),
                     "H1": H_clean["H1"], "H2": H_clean["H2"], "H3": H_clean["H3"],
                     "best_dist": (min(Z.items(), key=lambda kv: abs(kv[1]))[0] if Z else None)})

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
with open(OUT_REPORT, "w") as fh:
    fh.write("RFA STAGE 3 REPORT\n" + "\n".join(report))

print("\n" + "="*70); print("SUMMARY"); print("="*70)
summ = pd.DataFrame(rows).groupby("region").agg(
    sites=("station", "count"), discordant=("discordant", "sum"),
    H1=("H1", "first"), H2=("H2", "first"), H3=("H3", "first"),
    dist=("best_dist", "first")).round(2)
print(summ.to_string())
print(f"\nSaved: {OUT_CSV}\nSaved: {OUT_REPORT}")
print("\nIf any H1 >= 2: remove the discordant sites, move them to a neighbouring")
print("region in PC space, or lower k in Stage 2 and re-run this stage.")