"""
STEP 2 - REGION OF INFLUENCE (ROI) FOR EVERY CATCHMENT
======================================================
Inputs (from Step 1, in <DATA_DIR>/step1_outputs/):
    step1_pc_scores_for_ROI.csv    attribute space (all PCs)
    step1_annual_max_series.csv    annual maximum streamflow per station
    step1_final_attributes.csv     lat/lon for maps

For EACH target catchment, independently:
  1. Start with the target catchment as the region.
  2. Add stations one at a time. GROWTH_MODE = "centroid" (default): the region is treated as ONE
     combined point = mean PC score of its current members; the next station is the one with the
     smallest Euclidean distance (all PCs) to that point. [1] -> [1,2] -> point "12" -> nearest to
     "12" is added -> new combined point -> ...   ("target" = classical ROI, distance to target only)
     - Discordancy Di (Hosking & Wallis 1997, eq 3.3) is computed on the L-moment ratios
       (L-CV, L-skew, L-kurt) of the candidate region; a newly added station that is
       discordant is EXCLUDED and recorded, and growth continues with the next neighbour.
     - From ROI_MIN_SIZE stations onward, heterogeneity H1, H2, H3 are computed from the
       annual-maximum STREAMFLOW L-moments (kappa-distribution Monte Carlo, H&W 1997 s4.3).
  3. Region keeps growing while H1 < 1. Growth stops after ROI_PATIENCE consecutive
     additions with H1 >= 2 (or at ROI_MAX_SIZE); the final region is the largest one
     before H1 first stays >= 1 (tier "H1<1"). If no size reaches H1 < 1, the largest
     size with H1 < 2 is used (tier "1<=H1<2 only"); otherwise the minimum size is
     reported as heterogeneous.
  4. Final region is re-tested with NSIM_FINAL simulations; H1/H2/H3 and Di reported.

Outputs in <DATA_DIR>/step2_outputs/
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
IN_DIR = "step1_outputs"
OUT_NAME = "step2_outputs"

GROWTH_MODE = "centroid"    # "centroid": after each addition the region becomes ONE combined point
                            #   (mean of its members' PC scores); the next station added is the one
                            #   nearest to that combined point  [1] -> [1,2] -> centroid(12) -> ...
                            # "target": classical ROI (Burn 1990), always nearest to the target catchment
ROI_MIN_SIZE = 5            # H not evaluated below this many stations (H is meaningless for 2-4)
ROI_MAX_SIZE = 60
ROI_PATIENCE = 2            # stop after this many consecutive H1 >= 2 (H1 is noisy)
ROI_PATIENCE_GRACE = 10     # patience-based stopping is only allowed once the region has reached
                            # this size. H1 at N = 5-7 is highly sample-noisy (H&W 1997 s4.3 warn
                            # it is "meaningless" there); judging growth on it at that size means
                            # almost every region gets 2 noisy draws and quits at the floor before
                            # ever being tested at a size where H1 is a reliable statistic.
EXCLUDE_DISCORDANT = True   # drop a newly added neighbour if it is discordant in the region
DI_MIN_REGION = 7           # Di exclusion only once the trial region has >= 7 stations:
                            # for N = 5-6 the largest POSSIBLE Di is (N-1)/3, which equals the
                            # critical value, so the test is degenerate there
MAX_CONSEC_EXCLUSIONS = 10  # safety: after this many exclusions in a row, stop excluding
DROP_DUPLICATE_SERIES = True  # stations with an identical streamflow series = same gauge twice
NSIM_GROW = 200             # simulations per H evaluation while growing (speed)
NSIM_FINAL = 500            # simulations for the reported H of the final region (H&W use 500)
SEED = 20260924             # fixed -> reproducible H, and smooth H-vs-size curves

DI_CRIT = {5: 1.333, 6: 1.648, 7: 1.917, 8: 2.140, 9: 2.329, 10: 2.491,
           11: 2.632, 12: 2.757, 13: 2.869, 14: 2.971}      # H&W 1997 Table 3.1; 3.0 for N >= 15


# =============================================================================
# L-MOMENTS
# =============================================================================
def _pwm_weights(n):
    j = np.arange(1, n + 1, dtype=float)
    w1 = (j - 1) / (n - 1)
    w2 = w1 * (j - 2) / (n - 2)
    w3 = w2 * (j - 3) / (n - 3)
    return w1 / n, w2 / n, w3 / n


def samlmom(x):
    """Unbiased sample L-moments -> l1, l2, t (L-CV), t3 (L-skew), t4 (L-kurt)."""
    x = np.sort(np.asarray(x, float))
    w1, w2, w3 = _pwm_weights(len(x))
    b0, b1, b2, b3 = x.mean(), x @ w1, x @ w2, x @ w3
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return b0, l2, l2 / b0, l3 / l2, l4 / l2


def lmr_matrix(X):
    X = np.sort(X, axis=1)
    w1, w2, w3 = _pwm_weights(X.shape[1])
    b0, b1, b2, b3 = X.mean(1), X @ w1, X @ w2, X @ w3
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l2 / b0, l3 / l2, l4 / l2


# =============================================================================
# KAPPA DISTRIBUTION (Hosking 1994) for the heterogeneity simulations
# =============================================================================
def _nudge(k):
    return k if abs(k) >= 1e-4 else (1e-4 if k >= 0 else -1e-4)


def _g(k, h):
    r = np.arange(1, 5, dtype=float)
    if abs(h) < 1e-6:
        return np.exp(gammaln(1 + k) - k * np.log(r))
    if h > 0:
        return r * np.exp(gammaln(1 + k) + gammaln(r / h) - (1 + k) * np.log(h) - gammaln(1 + k + r / h))
    return r * np.exp(gammaln(1 + k) + gammaln(-k - r / h) - (1 + k) * np.log(-h) - gammaln(1 - r / h))


def kappa_tau34(k, h):
    g = _g(_nudge(k), h)
    d = g[0] - g[1]
    return (-g[0] + 3 * g[1] - 2 * g[2]) / d, (g[0] - 6 * g[1] + 10 * g[2] - 5 * g[3]) / d, g


def fit_kappa(t3, t4):
    if t4 >= (1 + 5 * t3 ** 2) / 6:                    # above GLO line: kappa cannot fit
        return None
    c = 2 / (3 + t3) - np.log(2) / np.log(3)
    k0 = 7.8590 * c + 2.9554 * c ** 2

    def res(p):
        k, h = p
        if not (k > -1 and (h >= 0 or h * k > -1)):
            return [10.0, 10.0]
        a, b, _ = kappa_tau34(k, h)
        return [a - t3, b - t4] if np.isfinite(a) and np.isfinite(b) else [10.0, 10.0]

    best = None
    for h0 in (0.0, 0.5, -0.5, 1.0, -0.9, 2.0):
        try:
            sol = least_squares(res, [k0, h0], bounds=([-0.999, -1.0], [20.0, 50.0]),
                                xtol=1e-12, ftol=1e-12)
        except Exception:
            continue
        err = np.max(np.abs(sol.fun))
        if best is None or err < best[0]:
            best = (err, sol.x)
        if err < 1e-6:
            break
    return None if best is None or best[0] > 1e-4 else tuple(best[1])


class RegionalKappa:
    """Kappa with mean 1 and regional L-CV, L-skew, L-kurt (GLO if kappa cannot fit)."""

    def __init__(self, tR, t3R, t4R):
        fit = fit_kappa(t3R, t4R)
        if fit is None:
            k, h, self.dist = -t3R, -1.0, "GLO"
        else:
            (k, h), self.dist = fit, "KAP"
        self.k, self.h, self.tR = _nudge(k), h, tR
        _, _, g = kappa_tau34(self.k, h)
        self.mean_y = (1 - g[0]) / self.k
        self.l2_y = (g[0] - g[1]) / self.k

    def sample(self, U):
        F = np.clip(U, 1e-12, 1 - 1e-12)
        z = -np.log(F) if abs(self.h) < 1e-6 else (1 - F ** self.h) / self.h
        y = (1 - z ** self.k) / self.k
        return 1 + self.tR * (y - self.mean_y) / self.l2_y


# =============================================================================
# HETEROGENEITY (H1, H2, H3) AND DISCORDANCY (Di)
# =============================================================================
def _V(T, T3, T4, n):
    W = n / n.sum()
    d1 = T - (T @ W)[:, None]
    d3 = T3 - (T3 @ W)[:, None]
    d4 = T4 - (T4 @ W)[:, None]
    return np.vstack([np.sqrt((d1 ** 2) @ W),                 # V1: L-CV
                      np.sqrt(d1 ** 2 + d3 ** 2) @ W,         # V2: L-CV & L-skew
                      np.sqrt(d3 ** 2 + d4 ** 2) @ W])        # V3: L-skew & L-kurt


def heterogeneity(L, n, nsim, seed=SEED):
    """L: (N,3) site (t, t3, t4); n: record lengths. Returns (array[H1,H2,H3], sim distribution)."""
    L = np.asarray(L, float)
    n = np.asarray(n, float)
    N = len(n)
    W = n / n.sum()
    tR, t3R, t4R = W @ L
    Vobs = _V(L[:, 0][None], L[:, 1][None], L[:, 2][None], n)[:, 0]
    kap = RegionalKappa(tR, t3R, t4R)
    rng = np.random.default_rng(seed)
    T, T3, T4 = (np.empty((nsim, N)) for _ in range(3))
    for i in range(N):
        T[:, i], T3[:, i], T4[:, i] = lmr_matrix(kap.sample(rng.random((nsim, int(n[i])))))
    Vs = _V(T, T3, T4, n)
    return (Vobs - Vs.mean(1)) / Vs.std(1, ddof=1), kap.dist


def discordancy(L):
    U = np.asarray(L, float)
    N = len(U)
    if N < 5:
        return np.full(N, np.nan), np.nan
    D = U - U.mean(0)
    Di = N / 3 * np.einsum("ij,jk,ik->i", D, np.linalg.pinv(D.T @ D), D)
    return Di, DI_CRIT.get(N, 3.0)


def h_class(h1):
    return "acceptably homogeneous" if h1 < 1 else (
        "possibly heterogeneous" if h1 < 2 else "definitely heterogeneous")


# =============================================================================
# ROI GROWTH
# =============================================================================
def grow(t, Z, L, n):
    """Grow the region around target t. Returns region (in order of addition),
    excluded [(idx, size_at_exclusion, Di, crit)], H path, and a log of every step."""
    region, excluded, path, steps = [t], [], [], []
    pool = [j for j in range(len(Z)) if j != t]
    run = streak = 0
    while pool and len(region) < ROI_MAX_SIZE:
        ref = Z[region].mean(0) if GROWTH_MODE == "centroid" else Z[t]
        d_ref = np.linalg.norm(Z[pool] - ref, axis=1)
        c = pool.pop(int(np.argmin(d_ref)))          # nearest to the combined point (or target)
        d_c = float(d_ref.min())
        trial = region + [c]
        if EXCLUDE_DISCORDANT and len(trial) >= DI_MIN_REGION and streak < MAX_CONSEC_EXCLUSIONS:
            Di, crit = discordancy(L[trial])
            if Di[-1] > crit + 1e-9:
                excluded.append((c, len(region), Di[-1], crit))
                steps.append(dict(size=len(region), candidate=c, dist_to_ref=d_c,
                                  dist_to_target=float(np.linalg.norm(Z[c] - Z[t])),
                                  action=f"excluded (Di={Di[-1]:.2f} > {crit})"))
                streak += 1
                continue
        streak = 0
        region = trial
        rec = dict(size=len(region), candidate=c, dist_to_ref=d_c,
                   dist_to_target=float(np.linalg.norm(Z[c] - Z[t])), action="added")
        if len(region) >= ROI_MIN_SIZE:
            H, _ = heterogeneity(L[region], n[region], NSIM_GROW)
            path.append(dict(size=len(region), added=c, H1=H[0], H2=H[1], H3=H[2]))
            rec.update(H1=H[0], H2=H[1], H3=H[2])
            run = run + 1 if (H[0] >= 2 and len(region) >= ROI_PATIENCE_GRACE) else 0
        steps.append(rec)
        if run >= ROI_PATIENCE:
            break
    return region, excluded, path, steps


def choose(path, thr):
    last_ok, run = None, 0
    for p in path:
        if p["H1"] < thr:
            last_ok, run = p, 0
        else:
            run = run + 1 if p["size"] >= ROI_PATIENCE_GRACE else 0
            if run >= ROI_PATIENCE:
                break
    return last_ok


# =============================================================================
def neighbour_similarity(D, t, k=5, reps=2000):
    """Does closeness in PC space mean similar flood behaviour (L-CV)? If not, ROI regions
    cannot become homogeneous whatever the growth rule."""
    from scipy.stats import spearmanr
    iu = np.triu_indices(len(t), 1)
    rho = spearmanr(D[iu], np.abs(t[:, None] - t[None, :])[iu])[0]
    nn = np.median([np.std(t[np.argsort(D[i])[:k]]) for i in range(len(t))])
    rng = np.random.default_rng(SEED)
    rnd = np.median([np.std(t[rng.choice(len(t), k, replace=False)]) for _ in range(reps)])
    print(f"Attribute-space check: Spearman(distance, |L-CV difference|) = {rho:.3f}  "
          f"(near 0 = attributes do not predict flood behaviour)")
    print(f"  L-CV spread of {k} nearest neighbours {nn:.3f} vs {k} random stations {rnd:.3f} "
          f"({100 * (1 - nn / rnd):.0f} % better than random)\n")


def verify_distances(D, names, src, n_pcs):
    """Independent check: with all PCs kept, PC-space distance must equal the distance
    between z-scored (transformed) attributes. Also checks symmetry / zero diagonal."""
    print("\nDistance checks:")
    print(f"  symmetric: {np.allclose(D, D.T)} | zero diagonal: {np.allclose(np.diag(D), 0)}")
    try:
        from sklearn.preprocessing import StandardScaler
        summ = pd.read_csv(os.path.join(src, "step1_attribute_summary.csv"), index_col=0)
        fa = pd.read_csv(os.path.join(src, "step1_final_attributes.csv"))
        X = fa[summ.index].astype(float).copy()          # same station set as the PCA
        for c, tr in summ["transform"].fillna("").items():
            if tr == "log":
                X[c] = np.log(X[c].clip(lower=1e-6))
        Xz = pd.DataFrame(StandardScaler().fit_transform(X.values), index=fa["station"].astype(str))
        Xz = Xz.loc[names].values
        Dz = cdist(Xz, Xz)
        err = np.abs(D - Dz).max()
        if n_pcs == len(summ):
            print(f"  PC-space distance vs z-scored attribute distance: max difference = {err:.2e} "
                  f"({'OK' if err < 1e-6 else 'MISMATCH - check Step 1 outputs'})")
        else:
            print(f"  using {n_pcs} of {len(summ)} PCs (retained set): distances differ from full "
                  f"attribute distance by up to {err:.3f}, as expected")
    except Exception as e:
        print(f"  (could not run attribute cross-check: {e})")
    print()


def main():
    src = os.path.join(DATA_DIR, IN_DIR)
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)

    pcs = pd.read_csv(os.path.join(src, "step1_pc_scores_for_ROI.csv"))
    ams = pd.read_csv(os.path.join(src, "step1_annual_max_series.csv"))
    attr = pd.read_csv(os.path.join(src, "step1_final_attributes.csv"))
    if DROP_DUPLICATE_SERIES:
        sig = ams.sort_values(["station", "year"]).groupby("station").apply(
            lambda g: tuple(zip(g["year"], g["ams"].round(6))), include_groups=False)
        dups = sig[sig.duplicated(keep="first")].index.astype(str)
        if len(dups):
            print(f"WARNING {len(dups)} station(s) have a streamflow series identical to another station "
                  f"(same gauge counted twice) and are dropped: {list(dups)}")
            pd.DataFrame({"dropped_station": dups, "reason": "identical streamflow series"}).to_csv(
                os.path.join(DATA_DIR, OUT_NAME, "step2_dropped_duplicates.csv"), index=False)
            pcs = pcs[~pcs["station"].astype(str).isin(dups)].reset_index(drop=True)
    names = pcs["station"].astype(str).values
    Z = pcs.drop(columns="station").values
    print(f"{len(names)} catchments, {Z.shape[1]} PCs as attribute space")

    # at-site L-moments of the annual-maximum streamflow
    rows = []
    for st in names:
        x = ams.loc[ams["station"].astype(str) == st, "ams"].values
        l1, l2, t, t3, t4 = samlmom(x)
        rows.append(dict(station=st, n_years=len(x), mean_ams=l1, L_CV=t, L_skew=t3, L_kurt=t4))
    lm = pd.DataFrame(rows)
    Dg, cg = discordancy(lm[["L_CV", "L_skew", "L_kurt"]].values)
    lm["Di_all_stations"], lm["discordant_all_stations"] = Dg, Dg > cg
    lm.to_csv(f"{out}/step2_atsite_lmoments.csv", index=False)
    L = lm[["L_CV", "L_skew", "L_kurt"]].values
    n = lm["n_years"].values.astype(float)
    print(f"{int((Dg > cg).sum())} stations discordant against the whole dataset (Di > {cg})")

    D = cdist(Z, Z)                                   # Euclidean distance in PC space
    pd.DataFrame(D, index=names, columns=names).to_csv(f"{out}/step2_distance_matrix.csv")
    verify_distances(D, names, src, Z.shape[1])
    neighbour_similarity(D, L[:, 0])
    nn = []
    for t in range(len(names)):
        for rank, j in enumerate([j for j in np.argsort(D[t]) if j != t][:15], start=1):
            nn.append(dict(station=names[t], rank=rank, neighbour=names[j], distance=D[t, j]))
    pd.DataFrame(nn).to_csv(f"{out}/step2_nearest_15_neighbours.csv", index=False)

    results, paths, members_long, excl_long = [], [], [], []
    t0 = time.time()
    for t in range(len(names)):
        region, excluded, path, steps = grow(t, Z, L, n)
        for st in steps:
            paths.append(dict(target=names[t], region_size=st["size"], candidate=names[st["candidate"]],
                              action=st["action"], dist_to_region_point=st["dist_to_ref"],
                              dist_to_target=st["dist_to_target"],
                              H1=st.get("H1", np.nan), H2=st.get("H2", np.nan), H3=st.get("H3", np.nan)))
        ok1, ok2 = choose(path, 1.0), choose(path, 2.0)
        if ok1:
            size, tier = ok1["size"], "H1<1"
        elif ok2:
            size, tier = ok2["size"], "1<=H1<2 only"
        else:
            size, tier = min(ROI_MIN_SIZE, len(region)), "heterogeneous at minimum size"
        final = region[:size]

        H, dist = heterogeneity(L[final], n[final], NSIM_FINAL)
        Di, crit = discordancy(L[final])
        disc_final = [names[j] for j, d in zip(final, Di)
                  if np.isfinite(d) and d > crit + 1e-9 and len(final) >= DI_MIN_REGION]
        excl = [e for e in excluded if e[1] < size]

        results.append(dict(
            station=names[t], roi_size=size, acceptance=tier,
            H1=H[0], H2=H[1], H3=H[2], H1_class=h_class(H[0]), sim_distribution=dist,
            pooled_years=int(n[final].sum()),
            regional_L_CV=float((n[final] / n[final].sum()) @ L[final, 0]),
            max_distance_from_target=float(D[t, final].max()),
            centroid_drift=float(np.linalg.norm(Z[final].mean(0) - Z[t])),
            n_discordant_excluded=len(excl),
            discordant_excluded=";".join(names[e[0]] for e in excl),
            Di_crit_final=crit, discordant_in_final=";".join(disc_final),
            target_discordant=names[t] in disc_final,
            members=";".join(names[j] for j in final)))
        for rank, j in enumerate(final):
            members_long.append(dict(target=names[t], order_added=rank, member=names[j],
                                     distance_from_target=D[t, j],
                                     Di=Di[rank] if np.isfinite(Di[rank]) else np.nan))
        for c, at, d, cr in excl:
            excl_long.append(dict(target=names[t], excluded=names[c], region_size_at_exclusion=at,
                                  Di=d, Di_crit=cr))
        if (t + 1) % 10 == 0 or t == len(names) - 1:
            el = time.time() - t0
            print(f"  {t + 1}/{len(names)} done  ({el / 60:.1f} min, ~{el / (t + 1) * (len(names) - t - 1) / 60:.1f} min left)")

    res = pd.DataFrame(results)
    res.to_csv(f"{out}/step2_roi_results.csv", index=False)
    pd.DataFrame(paths).to_csv(f"{out}/step2_roi_growth_paths.csv", index=False)
    pd.DataFrame(members_long).to_csv(f"{out}/step2_roi_members.csv", index=False)
    pd.DataFrame(excl_long, columns=["target", "excluded", "region_size_at_exclusion", "Di", "Di_crit"]
                 ).to_csv(f"{out}/step2_roi_discordant_excluded.csv", index=False)

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(res.roi_size, bins=range(ROI_MIN_SIZE, ROI_MAX_SIZE + 2), color="#4c72b0")
    ax[0].set_xlabel("ROI size (stations)"); ax[0].set_ylabel("catchments"); ax[0].set_title("ROI sizes")
    ax[1].hist(res.H1, bins=30, color="#55a868"); ax[1].axvline(1, c="k"); ax[1].axvline(2, c="k", ls=":")
    ax[1].set_xlabel("H1 of final ROI"); ax[1].set_title("H1 of final ROI regions")
    res.acceptance.value_counts().plot.bar(ax=ax[2], color="#c44e52", rot=15)
    ax[2].set_title("Acceptance tier")
    fig.tight_layout(); fig.savefig(f"{out}/step2_summary.png", dpi=200); plt.close(fig)

    P = pd.DataFrame(paths)
    P = P[(P.action == "added") & P.H1.notna()].rename(columns={"region_size": "size"}) if not P.empty else P
    if not P.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for st in res.sort_values("roi_size")["station"].iloc[:: max(1, len(res) // 12)]:
            p = P[P.target == st]
            ax.plot(p["size"], p["H1"], "-", lw=1, label=st)
        ax.axhline(1, c="k"); ax.axhline(2, c="k", ls=":")
        ax.set_xlabel("region size"); ax.set_ylabel("H1"); ax.set_title("H1 as ROI grows (sample of catchments)")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        fig.savefig(f"{out}/step2_H1_growth_curves.png", dpi=200); plt.close(fig)

    geo = attr.set_index(attr["station"].astype(str)).reindex(res.station)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7.5))
    for a, col, cmap, lab in ((ax[0], res.roi_size, "viridis", "ROI size"),
                              (ax[1], res.H1.clip(-2, 4), "RdYlGn_r", "H1 (clipped -2..4)")):
        s = a.scatter(geo["lon"], geo["lat"], c=col, cmap=cmap, s=30, edgecolor="k", lw=0.3)
        fig.colorbar(s, ax=a, label=lab)
        a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)"); a.set_title(lab)
        a.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(geo["lat"])))); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/step2_roi_map.png", dpi=200); plt.close(fig)

    # ---------------- summary ----------------
    print("\n================ STEP 2 SUMMARY ================")
    print(f"Growth mode: {GROWTH_MODE} | homogeneity: H1 < 1 acceptably homogeneous, "
          f"1 <= H1 < 2 possibly heterogeneous, H1 >= 2 definitely heterogeneous (Hosking & Wallis 1997)")
    print(res.acceptance.value_counts().to_string())
    print(f"\nROI size: median {res.roi_size.median():.0f}, range {res.roi_size.min()}-{res.roi_size.max()}")
    print(f"H1 of final regions: median {res.H1.median():.2f}")
    print(res.H1_class.value_counts().to_string())
    print(f"Catchments where a neighbour was excluded as discordant: {(res.n_discordant_excluded > 0).sum()}")
    print(f"Catchments whose own station is discordant in its ROI: {int(res.target_discordant.sum())}")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()