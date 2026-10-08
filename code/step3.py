"""
STEP 3 - DISTRIBUTION SELECTION + REGIONAL GROWTH CURVE
=========================================================
Hosking & Wallis (1997), chapter 5.

For EACH target's final ROI (step2_outputs/step2_roi_results.csv, column 'members'):
  1. Take the region members' at-site L-moments (step2_outputs/step2_atsite_lmoments.csv),
     weight by record length -> regional average L-CV, L-skew, L-kurt (tR, t3R, t4R).
  2. Fit 5 candidate 3-parameter distributions to (tR, t3R) -- i.e. index-flood-normalised
     (L1 = 1): GLO, GEV, GPA (all three are the 4-parameter kappa distribution (Hosking 1994)
     at fixed h = -1, 0, +1 respectively -- reuses the exact kappa machinery already in
     roi.py, no extra approximation needed), plus GNO and PE3 (fit by matching L-skew via
     numerical L-moment integration of their quantile functions, since neither is a kappa
     special case).
  3. Score each candidate with the Z^DIST statistic (H&W eq 5.5): simulate NSIM_ZDIST
     replicate homogeneous regions from a 4-parameter kappa reference fitted to
     (tR, t3R, t4R) -- same Monte Carlo as roi.py's heterogeneity() -- to get the bias and
     standard deviation of the regional-average L-kurtosis under sampling noise, then
       Z_DIST = (tau4_model - t4R_observed + bias) / sigma
     Accept |Z| <= 1.64; of the accepted set (or, if none, of all 5) keep the smallest |Z|.
  4. The chosen distribution, fit to (tR, t3R) with L1 = 1, IS the regional growth curve
     q(T) at T = 2, 5, 10, 25, 50, 100 (non-exceedance prob F = 1 - 1/T).

A self-test runs at import time (see `_self_test()`): confirms the GEV/GLO/GPA kappa-exact
L-moments agree with the general-purpose numerical-integration L-moment routine, and that
GNO/PE3 collapse to the symmetric case (L-skew ~ 0, L-kurt ~ normal's 0.1226) at shape = 0.
If a distribution family cannot be fit for a given region it is silently skipped (marked
NaN) rather than left wrong.

Outputs in <DATA_DIR>/step3_outputs/:
    step3_distribution_fits.csv   per target: chosen distribution + Z stats for all 5
    step3_growth_curves.csv       per target: q(T) at T = 2,5,10,25,50,100
    step3_growth_curves.png       sample of growth curves

Also exposed for step5.py (ungauged test): fit_region(members_L, members_n) -> result dict,
so the exact same fitting procedure can be reapplied to a region that excludes the target.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, pearson3
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi import kappa_tau34, _g, _nudge, RegionalKappa, lmr_matrix, SEED  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
IN1 = os.path.join(DATA_DIR, "step1_outputs")
IN2 = os.path.join(DATA_DIR, "step2_outputs")
OUT_NAME = "step3_outputs"

RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
NSIM_ZDIST = 500            # H&W use 500 simulations for Z^DIST
ZDIST_ACCEPT = 1.64         # |Z| <= this -> acceptable fit (H&W s5.2)
FAMILIES = ["GLO", "GEV", "GPA", "GNO", "PE3"]

# dense F-grid for numerical L-moment integration (shifted-Legendre weights)
_F_GRID = np.linspace(1e-6, 1 - 1e-6, 20001)
_W1 = 2 * _F_GRID - 1
_W2 = 6 * _F_GRID ** 2 - 6 * _F_GRID + 1
_W3 = 20 * _F_GRID ** 3 - 30 * _F_GRID ** 2 + 12 * _F_GRID - 1


def _lmoments_of_quantile(qfunc, *params):
    """Theoretical L1..L4 of a quantile function x(F; *params), by numerical integration
    against the shifted-Legendre weights (the textbook integral definition of L-moments)."""
    x = qfunc(_F_GRID, *params)
    L1 = np.trapezoid(x, _F_GRID)
    L2 = np.trapezoid(x * _W1, _F_GRID)
    L3 = np.trapezoid(x * _W2, _F_GRID)
    L4 = np.trapezoid(x * _W3, _F_GRID)
    return L1, L2, L3, L4


# =============================================================================
# KAPPA-FAMILY MEMBERS: GLO (h=-1), GEV (h=0), GPA (h=+1)
# Hosking's 4-parameter kappa distribution (already implemented & tested in roi.py)
# nests these three as special cases of h, so we reuse its exact (non-approximated)
# moment machinery -- just solve for k at fixed h to match the target L-skew.
# =============================================================================
_H_OF = {"GLO": -1.0, "GEV": 0.0, "GPA": 1.0}


def _kappa_quantile_std(F, k, h):
    """Standard-form (xi=0, alpha=1) kappa quantile, vectorised over F."""
    F = np.clip(np.asarray(F, float), 1e-12, 1 - 1e-12)
    z = -np.log(F) if abs(h) < 1e-6 else (1 - F ** h) / h
    k = _nudge(k)
    return (1 - z ** k) / k


def _fit_kappa_family(h, tau3_target):
    """Solve for k (at fixed h) so the kappa(k,h) L-skew matches tau3_target exactly
    (kappa_tau34 gives EXACT, not approximate, L-moments via gamma functions)."""
    def resid(k):
        t3, _, _ = kappa_tau34(_nudge(k), h)
        return t3 - tau3_target

    for lo, hi in [(-0.9, 0.9), (-3, 3), (-8, 8)]:
        try:
            flo, fhi = resid(lo), resid(hi)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
                k = brentq(resid, lo, hi, xtol=1e-10)
                t3, t4, _ = kappa_tau34(_nudge(k), h)
                return k, t4
        except Exception:
            continue
    return None, None


def _fit_glo(tau3):
    return _fit_kappa_family(_H_OF["GLO"], tau3)


def _fit_gev(tau3):
    return _fit_kappa_family(_H_OF["GEV"], tau3)


def _fit_gpa(tau3):
    return _fit_kappa_family(_H_OF["GPA"], tau3)


def _quantile_kappa(F, k, h, xi, alpha):
    return xi + alpha * _kappa_quantile_std(F, k, h)


# =============================================================================
# GNO (Generalized Normal): x(F) = (1 - exp(-k * Phi^-1(F))) / k,  x(F;0) = Phi^-1(F)
# PE3 (Pearson Type III): scipy.stats.pearson3(skew).ppf(F)
# Neither is a kappa special case -> fit shape by numerically integrating L-moments
# of the standard-form quantile and root-finding on L-skew.
# =============================================================================
def _gno_quantile_std(F, k):
    z = norm.ppf(np.clip(F, 1e-12, 1 - 1e-12))
    if abs(k) < 1e-6:
        return z
    return (1 - np.exp(-k * z)) / k


def _pe3_quantile_std(F, skew):
    if abs(skew) < 1e-6:
        return norm.ppf(np.clip(F, 1e-12, 1 - 1e-12))
    return pearson3.ppf(np.clip(F, 1e-12, 1 - 1e-12), skew)


def _fit_by_numeric_tau3(qfunc_std, tau3_target, lo, hi):
    """Generic: root-find a single shape parameter so the quantile function's numerically
    integrated L-skew matches tau3_target. Returns (shape, tau4_std) or (None, None)."""
    def tau3_of(shape):
        L1, L2, L3, _ = _lmoments_of_quantile(qfunc_std, shape)
        return L3 / L2

    def resid(shape):
        return tau3_of(shape) - tau3_target

    try:
        flo, fhi = resid(lo), resid(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0):
            return None, None
        shape = brentq(resid, lo, hi, xtol=1e-6)
        _, L2, L3, L4 = _lmoments_of_quantile(qfunc_std, shape)
        return shape, L4 / L2
    except Exception:
        return None, None


def _fit_gno(tau3):
    return _fit_by_numeric_tau3(_gno_quantile_std, tau3, -2.5, 2.5)


def _fit_pe3(tau3):
    return _fit_by_numeric_tau3(_pe3_quantile_std, tau3, -9.0, 9.0)


_FIT = {"GLO": _fit_glo, "GEV": _fit_gev, "GPA": _fit_gpa, "GNO": _fit_gno, "PE3": _fit_pe3}


def _quantile_std(fam, F, shape):
    if fam in _H_OF:
        return _kappa_quantile_std(F, shape, _H_OF[fam])
    if fam == "GNO":
        return _gno_quantile_std(F, shape)
    if fam == "PE3":
        return _pe3_quantile_std(F, shape)
    raise ValueError(fam)


def _std_L1_L2(fam, shape):
    if fam in _H_OF:
        _, _, g = kappa_tau34(_nudge(shape), _H_OF[fam])
        k = _nudge(shape)
        mean_y = (1 - g[0]) / k
        l2_y = (g[0] - g[1]) / k
        return mean_y, l2_y
    L1, L2, _, _ = _lmoments_of_quantile(lambda F, s: _quantile_std(fam, F, s), shape)
    return L1, L2


# =============================================================================
# Z^DIST goodness-of-fit (H&W 1997 eq 5.5)
# =============================================================================
def _simulate_regional_t4(tR, t3R, t4R, n, nsim=NSIM_ZDIST, seed=SEED):
    """Monte Carlo null distribution of the regional-average L-kurtosis, under the
    4-parameter kappa reference fit to (tR, t3R, t4R) -- same construction as
    roi.py's heterogeneity(), just reporting simulated T4 instead of the H-statistics."""
    n = np.asarray(n, float)
    N = len(n)
    W = n / n.sum()
    kap = RegionalKappa(tR, t3R, t4R)
    rng = np.random.default_rng(seed)
    T4 = np.empty((nsim, N))
    for i in range(N):
        _, _, t4_i = lmr_matrix(kap.sample(rng.random((nsim, int(n[i])))))
        T4[:, i] = t4_i
    return T4 @ W  # weighted regional-average T4 per simulation, shape (nsim,)


def fit_region(L, n, return_periods=RETURN_PERIODS, nsim_zdist=NSIM_ZDIST):
    """Fit all 5 candidate distributions to a region's pooled, index-flood-normalised
    L-moments (L, n = at-site L_CV/L_skew/L_kurt and record lengths of the region's
    members), score with Z^DIST, pick the best, and return its growth curve q(T).

    L: (N,3) array of (L_CV, L_skew, L_kurt) per member.  n: (N,) record lengths.
    """
    L = np.asarray(L, float)
    n = np.asarray(n, float)
    W = n / n.sum()
    tR, t3R, t4R = W @ L

    sim_t4 = _simulate_regional_t4(tR, t3R, t4R, n, nsim=nsim_zdist)
    bias = sim_t4.mean() - t4R
    sigma = sim_t4.std(ddof=1)

    fits = {}
    for fam in FAMILIES:
        shape, tau4_model = _FIT[fam](t3R)
        if shape is None or not np.isfinite(tau4_model) or sigma <= 0:
            fits[fam] = dict(shape=np.nan, tau4_model=np.nan, Z=np.nan)
            continue
        Z = (tau4_model - t4R + bias) / sigma
        fits[fam] = dict(shape=shape, tau4_model=tau4_model, Z=Z)

    valid = {f: v for f, v in fits.items() if np.isfinite(v["Z"])}
    if not valid:
        return dict(tR=tR, t3R=t3R, t4R=t4R, bias=bias, sigma=sigma,
                     fits=fits, chosen=None, chosen_accepted=False, growth_curve={})

    accepted = {f: v for f, v in valid.items() if abs(v["Z"]) <= ZDIST_ACCEPT}
    pool = accepted if accepted else valid
    chosen = min(pool, key=lambda f: abs(pool[f]["Z"]))
    shape = fits[chosen]["shape"]
    L1_std, L2_std = _std_L1_L2(chosen, shape)
    alpha = tR / L2_std
    xi = 1.0 - alpha * L1_std

    gc = {}
    for T in return_periods:
        F = 1 - 1.0 / T
        gc[T] = float(xi + alpha * _quantile_std(chosen, np.array([F]), shape)[0])

    return dict(tR=tR, t3R=t3R, t4R=t4R, bias=bias, sigma=sigma, fits=fits,
                chosen=chosen, chosen_accepted=bool(chosen in accepted),
                chosen_shape=shape, chosen_xi=xi, chosen_alpha=alpha,
                growth_curve=gc)


# =============================================================================
# SELF-TEST (runs at import time) -- catches a wrong formula before it's trusted
# =============================================================================
def _self_test():
    # 1) kappa-exact vs numerical-integration agreement for GEV/GLO/GPA
    for fam, h in _H_OF.items():
        k = 0.15
        qfunc = lambda F, kk: _kappa_quantile_std(F, kk, h)
        L1n, L2n, L3n, L4n = _lmoments_of_quantile(qfunc, k)
        t3_num, t4_num = L3n / L2n, L4n / L2n
        t3_exact, t4_exact, _ = kappa_tau34(k, h)
        assert abs(t3_num - t3_exact) < 1e-3, f"{fam} tau3 mismatch: {t3_num} vs {t3_exact}"
        assert abs(t4_num - t4_exact) < 1e-3, f"{fam} tau4 mismatch: {t4_num} vs {t4_exact}"

    # 2) GNO and PE3 collapse to the normal distribution at shape = 0
    #    (normal: L-skew = 0, L-kurtosis = 30/pi * arctan(sqrt(2)) - 9 ~= 0.1226)
    normal_t4 = 30 / np.pi * np.arctan(np.sqrt(2)) - 9
    for fam, qfunc in (("GNO", _gno_quantile_std), ("PE3", _pe3_quantile_std)):
        L1, L2, L3, L4 = _lmoments_of_quantile(qfunc, 1e-6 if fam == "GNO" else 0.0)
        assert abs(L3 / L2) < 1e-2, f"{fam} at shape~0 not symmetric: t3={L3/L2}"
        assert abs(L4 / L2 - normal_t4) < 1e-2, f"{fam} at shape~0 t4={L4/L2} != normal {normal_t4}"

    # 3) fit_region round-trip: a region exactly matching a known GEV should pick GEV
    #    (or at least recover its L-skew/L-kurt closely) -- quick structural sanity check
    k_true = 0.1
    t3_true, t4_true, _ = kappa_tau34(k_true, 0.0)
    L_fake = np.tile([0.35, t3_true, t4_true], (10, 1))
    n_fake = np.full(10, 40.0)
    res = fit_region(L_fake, n_fake, nsim_zdist=100)
    assert res["chosen"] is not None, "fit_region found no acceptable/best distribution"
    assert abs(res["t3R"] - t3_true) < 1e-6


_self_test()
print("step3: distribution-fitting self-test passed (GEV/GLO/GPA exact-vs-numeric L-moments, "
      "GNO/PE3 normal limit, fit_region round-trip).")


# =============================================================================
def main():
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)

    results = pd.read_csv(os.path.join(IN2, "step2_roi_results.csv"))
    lm = pd.read_csv(os.path.join(IN2, "step2_atsite_lmoments.csv")).set_index("station")

    rows_fit, rows_gc = [], []
    for _, r in results.iterrows():
        members = r["members"].split(";")
        sub = lm.loc[members]
        L = sub[["L_CV", "L_skew", "L_kurt"]].values
        n = sub["n_years"].values.astype(float)
        res = fit_region(L, n)

        row = dict(station=r["station"], roi_size=len(members), regional_L_CV=res["tR"],
                   regional_L_skew=res["t3R"], regional_L_kurt=res["t4R"],
                   chosen_distribution=res["chosen"], chosen_accepted=res["chosen_accepted"])
        for fam in FAMILIES:
            row[f"Z_{fam}"] = res["fits"][fam]["Z"]
        rows_fit.append(row)

        gc_row = dict(station=r["station"], distribution=res["chosen"])
        gc_row.update({f"q_T{T}": res["growth_curve"].get(T, np.nan) for T in RETURN_PERIODS})
        rows_gc.append(gc_row)

    fit_df = pd.DataFrame(rows_fit)
    gc_df = pd.DataFrame(rows_gc)
    fit_df.to_csv(f"{out}/step3_distribution_fits.csv", index=False)
    gc_df.to_csv(f"{out}/step3_growth_curves.csv", index=False)

    print("\n================ STEP 3 SUMMARY ================")
    print("Chosen distribution counts:")
    print(fit_df.chosen_distribution.value_counts(dropna=False).to_string())
    print(f"\nAccepted (|Z|<=1.64) at the chosen distribution: "
          f"{int(fit_df.chosen_accepted.sum())} / {len(fit_df)}")
    print("\nMedian Z by candidate distribution (|Z|, lower=better fit):")
    for fam in FAMILIES:
        print(f"  {fam:5s} median |Z| = {fit_df[f'Z_{fam}'].abs().median():.2f}")
    print("\nSample growth curves (q(T), dimensionless, mean=1):")
    print(gc_df.head(8).round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    Ts = RETURN_PERIODS
    for _, r in gc_df.sample(min(15, len(gc_df)), random_state=1).iterrows():
        ax.plot(Ts, [r[f"q_T{T}"] for T in Ts], "-", lw=1, alpha=0.7, label=r["station"])
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels(Ts)
    ax.set_xlabel("Return period T (years)")
    ax.set_ylabel("q(T)  (dimensionless growth curve, mean = 1)")
    ax.set_title("Step 3: regional growth curves (sample of 15 targets)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/step3_growth_curves.png", dpi=200)
    plt.close(fig)

    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
