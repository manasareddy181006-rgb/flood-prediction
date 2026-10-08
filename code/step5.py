"""
STEP 5 - THE "UNGAUGED SITE" TEST (full leave-one-out cross-validation)
==========================================================================
For EACH of the 90 gauged catchments in turn, pretend it has no streamflow record and
see what the pipeline would have predicted for it, using only the OTHER 89 sites:

  1. MASK the target's own annual-maximum record. Nothing downstream may use it.

  2. RE-DERIVE ITS REGION ("does it join the same stations as before?"): grow a pool of
     GAUGED NEIGHBOURS around the target in PC-attribute space (same nearest-centroid
     growth rule, discordancy exclusion and H1-based stopping as roi.py's grow()/choose())
     -- but Di and H1/H2/H3 are now computed on the POOL ONLY, never on the target, since
     an ungauged site contributes no flow data to those statistics. The target's attribute
     position is still used to search for neighbours (attributes are known; only the gauge
     record is hidden). Compared against the target's ORIGINAL region (step2_roi_results,
     where its own data WAS used) via membership overlap (Jaccard) and size.

  3. FIT THE GROWTH CURVE on that masked pool (step3.fit_region, unchanged procedure).

  4. PREDICT THE INDEX FLOOD mu_hat from attributes alone, via step4's MLR REFIT on the
     other 89 sites (forward selection rerun from scratch, so the target cannot leak into
     its own prediction through the choice of predictors either).

  5. Q_estimated(T) = mu_hat * q(T)  -- this "removes the normalisation" done in step 3
     (growth curves are index-flood-normalised, mean = 1; multiplying by mu_hat restores
     real discharge units) using ONLY information an ungauged site would actually have.

  6. Q_computed(T): the honest at-site answer -- step3.fit_region applied to the target's
     OWN real L-moments (region of size 1), scaled by its OWN real at-site mean. This is
     only possible because we secretly still have the data; it is the validation target,
     not part of the "ungauged" estimate.

  7. %% error(T) = (Q_estimated - Q_computed) / Q_computed * 100, at every return period;
     T = 50 and T = 100 are reported explicitly as requested, alongside the full curve.

Run over all 90 sites (not just one) -- the right way to validate a regional method is
the full leave-one-out distribution of errors, not a single site's result.

Outputs in <DATA_DIR>/step5_outputs/:
    step5_ungauged_test.csv        per station: region overlap, mu_hat vs mu_at_site,
                                    Q_estimated(T) vs Q_computed(T) and %% error for every T
    step5_region_overlap.csv       original vs re-derived ("ungauged") region membership
    step5_summary.txt              aggregate error statistics by return period
    step5_est_vs_computed_T50.png  Est vs Computed scatter, 1:1 line, T = 50
    step5_est_vs_computed_T100.png Est vs Computed scatter, 1:1 line, T = 100
    step5_error_by_return_period.png
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from roi import (GROWTH_MODE, ROI_MIN_SIZE, ROI_MAX_SIZE, ROI_PATIENCE, ROI_PATIENCE_GRACE,
                  DI_MIN_REGION, EXCLUDE_DISCORDANT, MAX_CONSEC_EXCLUSIONS, NSIM_GROW,
                  discordancy, heterogeneity, choose, SEED)  # noqa: E402
from step3 import fit_region, RETURN_PERIODS  # noqa: E402
import step4  # noqa: E402

# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
IN1 = os.path.join(DATA_DIR, "step1_outputs")
IN2 = os.path.join(DATA_DIR, "step2_outputs")
OUT_NAME = "step5_outputs"


# =============================================================================
# Ungauged growth: identical mechanics to roi.py's grow()/choose(), except Di/H1-H3
# are evaluated on the POOL (gauged neighbours) only -- the target index t is used for
# the growth-centroid distance but never enters L[...] / n[...] for the flow statistics.
# =============================================================================
def grow_ungauged(t, Z, L, n):
    pool, excluded, path = [], [], []
    candidates = [j for j in range(len(Z)) if j != t]
    run = streak = 0
    while candidates and len(pool) < ROI_MAX_SIZE:
        ref = Z[[t] + pool].mean(0) if GROWTH_MODE == "centroid" else Z[t]
        d_ref = np.linalg.norm(Z[candidates] - ref, axis=1)
        c = candidates.pop(int(np.argmin(d_ref)))
        trial = pool + [c]
        if EXCLUDE_DISCORDANT and len(trial) >= DI_MIN_REGION and streak < MAX_CONSEC_EXCLUSIONS:
            Di, crit = discordancy(L[trial])
            if Di[-1] > crit + 1e-9:
                excluded.append((c, len(pool)))
                streak += 1
                continue
        streak = 0
        pool = trial
        rec = dict(size=len(pool), candidate=c)
        if len(pool) >= ROI_MIN_SIZE:
            H, _ = heterogeneity(L[pool], n[pool], NSIM_GROW)
            rec.update(H1=H[0], H2=H[1], H3=H[2])
            path.append(rec)
            run = run + 1 if (H[0] >= 2 and len(pool) >= ROI_PATIENCE_GRACE) else 0
        if run >= ROI_PATIENCE:
            break
    return pool, excluded, path


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


# =============================================================================
def main():
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)

    pcs = pd.read_csv(os.path.join(IN1, "step1_pc_scores_for_ROI.csv"))
    names = pcs["station"].astype(str).values
    Z = pcs.drop(columns="station").values
    name_idx = {s: i for i, s in enumerate(names)}

    lm = pd.read_csv(os.path.join(IN2, "step2_atsite_lmoments.csv")).set_index("station")
    lm = lm.loc[names]
    L_all = lm[["L_CV", "L_skew", "L_kurt"]].values
    n_all = lm["n_years"].values.astype(float)
    mu_all = lm["mean_ams"].values.astype(float)

    attr = pd.read_csv(os.path.join(IN1, "step1_final_attributes.csv")).set_index("station")
    attr = attr.loc[names]

    orig = pd.read_csv(os.path.join(IN2, "step2_roi_results.csv")).set_index("station")

    rows, overlap_rows = [], []
    t0 = time.time()
    for idx, station in enumerate(names):
        t = name_idx[station]

        # --- step 2 re-done "ungauged" ---
        pool, excluded, path = grow_ungauged(t, Z, L_all, n_all)
        ok1, ok2 = choose(path, 1.0), choose(path, 2.0)
        if ok1:
            size, tier = ok1["size"], "H1<1"
        elif ok2:
            size, tier = ok2["size"], "1<=H1<2 only"
        else:
            size, tier = min(ROI_MIN_SIZE, len(pool)), "heterogeneous at minimum size"
        pool_final = pool[:size]
        pool_names = [names[j] for j in pool_final]

        orig_members = [s for s in orig.loc[station, "members"].split(";") if s != station]
        jac = jaccard(pool_names, orig_members)
        overlap_rows.append(dict(station=station, original_region_size=len(orig_members),
                                  ungauged_pool_size=len(pool_names), jaccard_overlap=jac,
                                  original_neighbours=";".join(orig_members),
                                  ungauged_neighbours=";".join(pool_names)))

        # --- step 3 re-done on the masked pool ---
        if len(pool_final) >= 1:
            res_pool = fit_region(L_all[pool_final], n_all[pool_final])
        else:
            res_pool = dict(chosen=None, growth_curve={})

        # --- step 4: MLR refit excluding this station, predict its mu ---
        model_loo = step4.fit_mlr(attr, lm, exclude_stations=[station])
        mu_hat = step4.predict_mu(model_loo, attr.loc[station])

        # --- "computed" (at-site, honest) answer ---
        res_site = fit_region(L_all[[t]], n_all[[t]])
        mu_true = mu_all[t]

        row = dict(station=station, original_region_size=len(orig_members),
                   ungauged_pool_size=len(pool_names), jaccard_overlap=jac,
                   region_distribution=res_pool["chosen"], atsite_distribution=res_site["chosen"],
                   mu_at_site=mu_true, mu_predicted=mu_hat,
                   mu_pct_error=100 * (mu_hat - mu_true) / mu_true if mu_true else np.nan,
                   mlr_predictors=";".join(model_loo["predictors"]))
        for T in RETURN_PERIODS:
            q_est = res_pool["growth_curve"].get(T, np.nan)
            q_comp = res_site["growth_curve"].get(T, np.nan)
            Q_est = mu_hat * q_est if np.isfinite(q_est) else np.nan
            Q_comp = mu_true * q_comp if np.isfinite(q_comp) else np.nan
            row[f"Q_est_T{T}"] = Q_est
            row[f"Q_computed_T{T}"] = Q_comp
            row[f"pct_error_T{T}"] = (100 * (Q_est - Q_comp) / Q_comp
                                       if (Q_comp and np.isfinite(Q_comp) and Q_comp != 0) else np.nan)
        rows.append(row)

        if (idx + 1) % 15 == 0 or idx == len(names) - 1:
            el = time.time() - t0
            print(f"  {idx + 1}/{len(names)} done ({el / 60:.1f} min, "
                  f"~{el / (idx + 1) * (len(names) - idx - 1) / 60:.1f} min left)")

    res_df = pd.DataFrame(rows)
    ov_df = pd.DataFrame(overlap_rows)
    res_df.to_csv(f"{out}/step5_ungauged_test.csv", index=False)
    ov_df.to_csv(f"{out}/step5_region_overlap.csv", index=False)

    # ---------------- summary ----------------
    lines = []
    lines.append("================ STEP 5 SUMMARY: UNGAUGED-SITE LEAVE-ONE-OUT TEST ================\n")
    lines.append(f"Stations tested: {len(res_df)}\n")
    lines.append(f"\nRegion membership stability (original vs re-derived with own data masked):")
    lines.append(f"  median Jaccard overlap = {ov_df.jaccard_overlap.median():.2f}")
    lines.append(f"  exact same neighbour set (Jaccard=1): {int((ov_df.jaccard_overlap == 1).sum())} / {len(ov_df)}")
    lines.append(f"  no overlap at all (Jaccard=0): {int((ov_df.jaccard_overlap == 0).sum())} / {len(ov_df)}\n")
    lines.append(f"Index flood (mu) prediction, |%% error| (leave-one-out MLR):")
    lines.append(f"  median {res_df.mu_pct_error.abs().median():.1f}%%, "
                 f"IQR [{res_df.mu_pct_error.abs().quantile(.25):.1f}, "
                 f"{res_df.mu_pct_error.abs().quantile(.75):.1f}]%%\n")
    lines.append("Q(T) estimated-vs-computed |%% error| by return period:")
    for T in RETURN_PERIODS:
        col = res_df[f"pct_error_T{T}"].abs()
        lines.append(f"  T={T:>4d} yr: median {col.median():6.1f}%%  "
                     f"IQR [{col.quantile(.25):6.1f}, {col.quantile(.75):6.1f}]%%  "
                     f"(n={col.notna().sum()})")
    summary = "\n".join(lines)
    with open(f"{out}/step5_summary.txt", "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)

    # ---------------- plots ----------------
    for T in (50, 100):
        est, comp = res_df[f"Q_est_T{T}"], res_df[f"Q_computed_T{T}"]
        ok = est.notna() & comp.notna()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(comp[ok], est[ok], s=20, alpha=0.7, c="#4c72b0", edgecolor="k", lw=0.3)
        lo, hi = np.nanmin([comp[ok].min(), est[ok].min()]), np.nanmax([comp[ok].max(), est[ok].max()])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Q_computed, T={T}yr (at-site, cumec)")
        ax.set_ylabel(f"Q_estimated, T={T}yr (ungauged pipeline, cumec)")
        ax.set_title(f"Ungauged-site test: Estimated vs Computed, T={T} years")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out}/step5_est_vs_computed_T{T}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [res_df[f"pct_error_T{T}"].dropna().clip(-200, 200) for T in RETURN_PERIODS]
    ax.boxplot(data, tick_labels=[str(T) for T in RETURN_PERIODS], showfliers=False)
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xlabel("Return period T (years)")
    ax.set_ylabel("%% error, Q_estimated vs Q_computed (clipped +-200%%)")
    ax.set_title("Ungauged-site test: error grows with return period")
    fig.tight_layout()
    fig.savefig(f"{out}/step5_error_by_return_period.png", dpi=200)
    plt.close(fig)

    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
