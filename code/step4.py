"""
STEP 4 - MULTIPLE LINEAR REGRESSION OF THE INDEX FLOOD (mu) ON THE PRINCIPAL COMPONENTS
==========================================================================================
Step 3 gives the dimensionless REGIONAL growth curve q(T) (mean = 1). To turn it into a
discharge at an UNGAUGED site, Q(T) = mu_hat * q(T), the index flood mu (at-site mean
annual maximum flow) must come from catchment attributes alone. This "removes the
normalisation" of Step 3.

METHOD
  1. Attributes -> PCs, exactly as Step 1: the 10 attributes of step1.ATTRS (area as ln),
     standardised  z_j = (x_j - mean_j) / sd_j,  then  PC_k = sum_j z_j * e_jk  where e_jk is
     the eigenvector loading of attribute j on PC k.
  2. Regression on the PCs:   log10(mu) = b0 + b1*PCa + b2*PCb + ...
     Log10 because mu spans ~3 orders of magnitude and flood size grows as a power of
     catchment size -- the relation is linear, with near-constant error variance, in logs.
  3. Only SIGNIFICANT PCs: forward stepwise selection -- each round adds the PC with the
     smallest p-value (t-test on its coefficient), as long as p < P_ENTER. PCs are mutually
     uncorrelated, so there is no collinearity between predictors.
  4. Coefficients b, standard errors, t, p, R^2, adjusted R^2; the PC equation is also
     re-expressed in the original attributes (substituting PC_k = sum_j z_j e_jk).
  5. Back-transform with Duan's smearing estimate: mu_hat = S * 10**(fitted log10 mu),
     S = mean(10**residual). A plain 10**(fit) is the median, not the mean, and under-predicts.

UNGAUGED SITE: its attributes are standardised with the TRAINING mean/sd and multiplied by
the TRAINING eigenvectors to get its PC scores (to_pcs), then put into the equation.
fit_mlr(..., exclude_stations=[s]) refits the PCA AND reruns the selection without s, so the
leave-one-out ("ungauged") estimate never sees s. step5.py uses the same functions.

Outputs in <DATA_DIR>/step4_outputs/:
    step4_indexflood_model.txt       equation, coefficient table, diagnostics, worked example
    step4_coefficients.csv           b, SE, t, p for each selected PC
    step4_attribute_equation.csv     the same equation in terms of the original attributes
    step4_mu_predictions.csv         observed mu vs estimated mu (in-sample and leave-one-out)
    step4_regression_line.png        observed vs fitted log10(mu) with the regression line
    step4_partial_regression_PCs.png one partial-regression plot per selected PC
    step4_obs_vs_est_mean.png        observed vs estimated mean (in-sample | leave-one-out)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step1 import ATTRS  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
IN1 = os.path.join(DATA_DIR, "step1_outputs")
IN2 = os.path.join(DATA_DIR, "step2_outputs")
OUT_NAME = "step4_outputs"

P_ENTER = 0.05              # a PC enters the regression only if its coefficient has p < this
EXAMPLE_STATION = "Muthankera"
BLUE, GREY = "#2a6fb0", "#7a7a7a"


# =============================================================================
# ATTRIBUTES -> PCs
# =============================================================================
def transformed_attrs(attr_df):
    """The 10 Step 1 attributes with Step 1's transforms (area -> ln)."""
    X = attr_df[list(ATTRS)].astype(float).copy()
    for c, tr in ATTRS.items():
        if tr == "log":
            X[c] = np.log(X[c].clip(lower=1e-6))
    return X


def pca_fit(attr_df):
    """Standardise + PCA on the given (training) stations. Same as step1.run_pca:
    StandardScaler (population sd) then sklearn PCA keeping all components."""
    X = transformed_attrs(attr_df)
    mean, sd = X.mean(), X.std(ddof=0)
    pca = PCA().fit(((X - mean) / sd).values)
    pcs = [f"PC{i + 1}" for i in range(pca.n_components_)]
    E = pd.DataFrame(pca.components_.T, index=X.columns, columns=pcs)   # eigenvectors e_jk
    return dict(mean=mean, sd=sd, E=E, centre=pca.mean_, eigenvalues=pca.explained_variance_)


def to_pcs(pm, attr_df):
    """PC scores of any station (gauged or ungauged) from its attributes:
    z = (x - mean_train) / sd_train,  PC = z @ eigenvectors."""
    z = (transformed_attrs(attr_df) - pm["mean"]) / pm["sd"]
    return pd.DataFrame((z.values - pm["centre"]) @ pm["E"].values,
                        index=attr_df.index, columns=pm["E"].columns)


# =============================================================================
# OLS + forward stepwise selection on p-values
# =============================================================================
def ols(X, y):
    """X: (n,p) without intercept. Full OLS table."""
    n = len(y)
    Xd = np.column_stack([np.ones(n), X])
    p = Xd.shape[1] - 1
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    fitted = Xd @ beta
    resid = y - fitted
    dof = n - p - 1
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)))
    t = beta / se
    pval = 2 * stats.t.sf(np.abs(t), dof)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (resid @ resid) / sst
    adj = 1 - (1 - r2) * (n - 1) / dof
    F = (r2 / p) / ((1 - r2) / dof) if p > 0 else np.nan
    return dict(beta=beta, se=se, t=t, p=pval, fitted=fitted, resid=resid, s=np.sqrt(s2),
                r2=r2, adj_r2=adj, F=F, F_p=stats.f.sf(F, p, dof) if p > 0 else np.nan, dof=dof)


def forward_stepwise(P, y, p_enter=P_ENTER):
    """Add, one at a time, the PC whose coefficient has the smallest p-value, while p < p_enter.
    Returns the chosen PCs in order of entry and the p-value at entry."""
    chosen, entry_p = [], []
    while True:
        best = None
        for c in P.columns.difference(chosen, sort=False):
            pv = ols(P[chosen + [c]].values, y)["p"][-1]
            if best is None or pv < best[1]:
                best = (c, pv)
        if best is None or best[1] >= p_enter:
            break
        chosen.append(best[0])
        entry_p.append(best[1])
    return chosen, entry_p


# =============================================================================
def fit_mlr(attr_df, lm_df, exclude_stations=None, p_enter=P_ENTER):
    """attr_df: indexed by station (Step 1 attributes). lm_df: indexed by station, 'mean_ams'.
    exclude_stations: dropped from training -- PCA AND selection are redone without them."""
    stations = attr_df.index.intersection(lm_df.index)
    if exclude_stations:
        stations = stations.difference(set(exclude_stations))
    pm = pca_fit(attr_df.loc[stations])
    P = to_pcs(pm, attr_df.loc[stations])
    y = np.log10(lm_df.loc[stations, "mean_ams"].astype(float).values)
    chosen, entry_p = forward_stepwise(P, y, p_enter)
    fit = ols(P[chosen].values, y)
    return dict(pca=pm, predictors=chosen, entry_p=entry_p, fit=fit, beta=fit["beta"],
                smearing=float(np.mean(10 ** fit["resid"])), n_train=len(stations),
                train_stations=list(stations), pc_scores=P, y=y)


def predict_log10_mu(model, attr_df):
    P = to_pcs(model["pca"], attr_df)[model["predictors"]].values
    return model["beta"][0] + P @ model["beta"][1:]


def predict_mu(model, attr_row):
    """attr_row: Series (one station) or DataFrame. Returns mu_hat in cumec."""
    df = attr_row.to_frame().T if isinstance(attr_row, pd.Series) else attr_row
    mu = model["smearing"] * 10 ** predict_log10_mu(model, df)
    return float(mu[0]) if isinstance(attr_row, pd.Series) else mu


def attribute_equation(model):
    """log10(mu) = b0 + sum_k b_k PC_k with PC_k = sum_j e_jk (x_j - m_j)/s_j
    => per-attribute coefficient c_j = sum_k b_k e_jk / s_j (per unit of x_j) and
       std_j = sum_k b_k e_jk (per 1 s.d. of x_j)."""
    pm, b = model["pca"], model["beta"][1:]
    E = pm["E"][model["predictors"]].values
    std = E @ b
    c = std / pm["sd"].values
    const = model["beta"][0] - std @ (pm["centre"]) - c @ pm["mean"].values
    units = [("ln(" + j + ")") if ATTRS[j] == "log" else j for j in pm["E"].index]
    return pd.DataFrame({"attribute": units, "coef_per_unit": c, "coef_per_1sd": std,
                         "mean_train": pm["mean"].values, "sd_train": pm["sd"].values}), const


# =============================================================================
def main():
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)

    attr = pd.read_csv(os.path.join(IN1, "step1_final_attributes.csv")).set_index("station")
    lm = pd.read_csv(os.path.join(IN2, "step2_atsite_lmoments.csv")).set_index("station")
    attr = attr.loc[lm.index]

    model = fit_mlr(attr, lm)
    fit, sel = model["fit"], model["predictors"]

    # the PCs used here must be Step 1's PCs
    s1 = pd.read_csv(os.path.join(IN1, "step1_pc_scores_all_components.csv")).set_index("station")
    pc_check = float(np.abs(model["pc_scores"].values - s1.loc[model["pc_scores"].index].values).max())

    # leave-one-out = ungauged estimate of every station
    mu_obs = lm["mean_ams"].astype(float)
    mu_in = pd.Series(predict_mu(model, attr), index=attr.index)
    mu_loo, lg_loo, loo_sel = {}, {}, {}
    for st in attr.index:
        m = fit_mlr(attr, lm, exclude_stations=[st])
        mu_loo[st] = predict_mu(m, attr.loc[st])
        lg_loo[st] = float(predict_log10_mu(m, attr.loc[[st]])[0])
        loo_sel[st] = ";".join(m["predictors"])
    mu_loo, lg_loo = pd.Series(mu_loo), pd.Series(lg_loo)

    pred = pd.DataFrame({"mu_observed": mu_obs, "mu_est_insample": mu_in, "mu_est_loo": mu_loo,
                         "log10_mu_observed": np.log10(mu_obs),
                         "log10_mu_fitted": predict_log10_mu(model, attr), "log10_mu_loo": lg_loo})
    pred["pct_error_insample"] = 100 * (pred.mu_est_insample - pred.mu_observed) / pred.mu_observed
    pred["pct_error_loo"] = 100 * (pred.mu_est_loo - pred.mu_observed) / pred.mu_observed
    pred["loo_selected_PCs"] = pd.Series(loo_sel)
    pred = pd.concat([pred, model["pc_scores"][sel]], axis=1)
    pred.index.name = "station"
    pred.to_csv(f"{out}/step4_mu_predictions.csv")

    coef = pd.DataFrame({"term": ["intercept"] + sel, "b": fit["beta"], "SE": fit["se"],
                         "t": fit["t"], "p": fit["p"],
                         "entry_order": [np.nan] + list(range(1, len(sel) + 1))})
    coef.to_csv(f"{out}/step4_coefficients.csv", index=False)
    aeq, aconst = attribute_equation(model)
    aeq.to_csv(f"{out}/step4_attribute_equation.csv", index=False)

    def r2_log(col):
        """R^2 of the log10 fit itself (before the smearing back-transform, which only
        shifts every estimate by the same constant)."""
        y, f = pred.log10_mu_observed, pred[col]
        return 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)

    # ---------------- report ----------------
    eqn = f"log10(mu) = {fit['beta'][0]:.4f}" + "".join(
        f" {'+' if b >= 0 else '-'} {abs(b):.4f}*{p}" for p, b in zip(sel, fit["beta"][1:]))
    L = ["STEP 4 - INDEX FLOOD (mu) REGRESSION ON PRINCIPAL COMPONENTS", "=" * 62, "",
         f"Training stations: {model['n_train']}   response: log10(mean annual maximum flow, cumec)",
         f"Candidate predictors: PC1..PC{len(model['pca']['E'].columns)} (Step 1 PCA, all components)",
         f"Selection: forward stepwise, a PC enters only if p < {P_ENTER}",
         f"PC scores reproduce step1_pc_scores_all_components.csv: max |diff| = {pc_check:.1e}", "",
         "Entry order (p at entry): " + ", ".join(f"{p} ({pv:.2g})" for p, pv in zip(sel, model["entry_p"])),
         "", eqn, "", "Coefficients:", coef.to_string(index=False, float_format=lambda v: f"{v:.4g}"), "",
         f"R^2 = {fit['r2']:.4f}   adjusted R^2 = {fit['adj_r2']:.4f}   "
         f"residual s.e. (log10) = {fit['s']:.4f}   F = {fit['F']:.2f} (p = {fit['F_p']:.2g})",
         f"Duan smearing factor S = {model['smearing']:.4f}   mu_hat = S * 10**(fitted log10 mu)", "",
         "Same equation in the ORIGINAL attributes (PC_k = sum_j e_jk (x_j - mean_j)/sd_j substituted):",
         f"  log10(mu) = {aconst:.4f} + sum_j coef_per_unit_j * x_j",
         aeq.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
         "  (coef_per_1sd = change in log10 mu for a 1-s.d. increase of that attribute)", "",
         "Observed vs estimated mu:",
         f"  in-sample     : R^2(log10) = {r2_log('log10_mu_fitted'):.3f}, median |%err| = {pred.pct_error_insample.abs().median():.1f}%",
         f"  leave-one-out : R^2(log10) = {r2_log('log10_mu_loo'):.3f}, median |%err| = {pred.pct_error_loo.abs().median():.1f}%"
         f"  (= the ungauged-site accuracy)",
         f"  leave-one-out selected the same PCs as the full model in "
         f"{int((pred.loo_selected_PCs == ';'.join(sel)).sum())}/{len(pred)} cases"]

    # worked example: converting one station to PCs like an ungauged site
    ex = EXAMPLE_STATION if EXAMPLE_STATION in attr.index else attr.index[0]
    m_ex = fit_mlr(attr, lm, exclude_stations=[ex])
    pm = m_ex["pca"]
    x = transformed_attrs(attr.loc[[ex]]).iloc[0]
    z = (x - pm["mean"]) / pm["sd"]
    pc = to_pcs(pm, attr.loc[[ex]]).iloc[0]
    L += ["", f"WORKED EXAMPLE - '{ex}' treated as UNGAUGED (PCA + regression refit without it)",
          f"  equation without it: log10(mu) = {m_ex['beta'][0]:.4f}" + "".join(
              f" {'+' if b >= 0 else '-'} {abs(b):.4f}*{p}" for p, b in zip(m_ex["predictors"], m_ex["beta"][1:])),
          "  attribute            x        mean      sd        z = (x-mean)/sd"]
    L += [f"  {j:18s} {x[j]:9.3f} {pm['mean'][j]:9.3f} {pm['sd'][j]:9.3f} {z[j]:8.3f}" for j in x.index]
    for p in m_ex["predictors"]:
        terms = " + ".join(f"({z[j]:.2f})({pm['E'].loc[j, p]:.3f})" for j in x.index)
        L.append(f"  {p} = sum z_j * e_j,{p} = {terms} = {pc[p]:.3f}")
    lg = m_ex["beta"][0] + sum(b * pc[p] for p, b in zip(m_ex["predictors"], m_ex["beta"][1:]))
    L += [f"  log10(mu) = {lg:.4f} -> mu_hat = {m_ex['smearing']:.3f} * 10^{lg:.4f} = "
          f"{m_ex['smearing'] * 10 ** lg:.1f} cumec   (observed mu = {mu_obs[ex]:.1f} cumec)"]
    report = "\n".join(L)
    with open(f"{out}/step4_indexflood_model.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)

    # ---------------- plots ----------------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})

    # 1) regression line: observed vs fitted log10(mu)
    yf, yo = pred.log10_mu_fitted, pred.log10_mu_observed
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lo, hi = min(yf.min(), yo.min()) - 0.2, max(yf.max(), yo.max()) + 0.2
    xx = np.array([lo, hi])
    ax.fill_between(xx, xx - fit["s"], xx + fit["s"], color=BLUE, alpha=0.10, lw=0,
                    label=f"±1 residual s.e. ({fit['s']:.2f})")
    ax.plot(xx, xx, color=BLUE, lw=2, label="regression line (observed = fitted)")
    ax.scatter(yf, yo, s=36, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("fitted log10(mu)  = " + " + ".join(["b0"] + [f"b·{p}" for p in sel]))
    ax.set_ylabel("observed log10(mu)  (mu in cumec)")
    ax.set_title(f"Step 4 regression on PCs (n = {model['n_train']})\n"
                 f"R² = {fit['r2']:.2f}, adj. R² = {fit['adj_r2']:.2f}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out}/step4_regression_line.png", dpi=200); plt.close(fig)

    # 2) partial regression (component + residual) per selected PC
    k = len(sel)
    ncol = min(k, 3)
    nrow = int(np.ceil(k / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.1 * nrow), squeeze=False)
    for i, p in enumerate(sel):
        a = axs[i // ncol, i % ncol]
        b = fit["beta"][1 + i]
        xv = model["pc_scores"][p].values
        yv = b * xv + fit["resid"]
        a.scatter(xv, yv, s=26, color=BLUE, edgecolor="white", lw=0.6)
        xs = np.linspace(xv.min(), xv.max(), 2)
        a.plot(xs, b * xs, color="black", lw=2)
        a.set_xlabel(f"{p} score")
        a.set_ylabel(f"b·{p} + residual   (log10 mu)")
        a.set_title(f"{p}: b = {b:+.3f}, p = {fit['p'][1 + i]:.1g}", fontsize=10)
    for j in range(k, nrow * ncol):
        axs[j // ncol, j % ncol].axis("off")
    fig.suptitle("Partial regression of log10(mu) on each selected PC (slope = its coefficient)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/step4_partial_regression_PCs.png", dpi=200); plt.close(fig)

    # 3) observed vs estimated mean
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    for a, col, lgc, ttl in ((axs[0], "mu_est_insample", "log10_mu_fitted", "In-sample (all 90 stations fitted)"),
                             (axs[1], "mu_est_loo", "log10_mu_loo", "Leave-one-out (station treated as ungauged)")):
        o, e = pred.mu_observed, pred[col]
        lo, hi = min(o.min(), e.min()) / 1.5, max(o.max(), e.max()) * 1.5
        a.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1.2, label="1:1")
        a.scatter(o, e, s=36, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
        for st in pred.index[np.abs(np.log10(e / o)) > 1.2]:          # > ~16x off: name it
            a.annotate(st, (o[st], e[st]), xytext=(5, 3), textcoords="offset points", fontsize=8)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_aspect("equal")
        a.set_xlabel("observed mean annual max flow, mu (cumec)")
        a.set_ylabel("estimated mu from regression (cumec)")
        err = pred[col.replace("mu_est", "pct_error")].abs().median()
        a.set_title(f"{ttl}\nR²(log) = {r2_log(lgc):.2f}, median |error| = {err:.0f}%", fontsize=10)
        a.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Observed vs estimated mean (index flood)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/step4_obs_vs_est_mean.png", dpi=200); plt.close(fig)

    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
