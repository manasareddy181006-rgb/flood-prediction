"""
RFA STAGE 2 — Transform, standardize, PCA, and clustering into candidate regions
=================================================================================

Input : rfa_attributes.csv  (from Stage 1)
Output: rfa_regions.csv     station -> cluster assignment + PC scores
        rfa_pca_report.txt  loadings, variance explained, cluster profiles
        rfa_diagnostics.png scree / dendrogram / silhouette / cluster map

METHOD (Rao & Srinivas 2006 hybrid)
  1. Log-transform skewed attributes (area, stream length, rainfall) so a handful
     of huge basins don't dominate every Euclidean distance.
  2. Z-score standardize.
  3. PCA -> retain PCs to ~85% cumulative variance. Attributes here are heavily
     correlated (elev/temp/rain/slope move together); PCs are orthogonal so
     distance is meaningful.
  4. Ward's hierarchical clustering to initialise centroids, then K-means refine.
  5. Choose K by silhouette + the constraint that every region must hold enough
     CONTRIBUTING sites to be testable for homogeneity.

IMPORTANT
  Clustering uses ALL 276 target sites, because regions must be definable for
  ungauged catchments. The homogeneity test in Stage 3 only uses the contributing
  subset within each region.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# CONFIG
# =============================================================================
BASE = r"C:\Users\bsnre\Downloads\chingka kalai"

IN_ATTRIBUTES = os.path.join(BASE, "rfa_attributes.csv")
OUT_REGIONS   = os.path.join(BASE, "rfa_regions.csv")
OUT_REPORT    = os.path.join(BASE, "rfa_pca_report.txt")
OUT_FIG       = os.path.join(BASE, "rfa_diagnostics.png")

VAR_TARGET   = 0.85   # cumulative variance to retain
K_RANGE      = range(3, 9)
MIN_CONTRIB_PER_REGION = 5    # a region with fewer cannot be homogeneity-tested
INCLUDE_LATLON = False        # see note in the report; run both ways
RANDOM_STATE = 42

LOG_ATTRS = ["area_km2", "rain_max_mm", "rain_min_mm", "rain_annual_mean_mm", "relief_m"]


# =============================================================================
# 1. LOAD + SELECT ATTRIBUTES
# =============================================================================
print("=" * 70); print("1. LOAD"); print("=" * 70)

df = pd.read_csv(IN_ATTRIBUTES)
print(f"Sites: {len(df)}   Contributing: {int(df['is_contributing'].sum())}")

ATTRS = ["area_km2", "mean_elev_m", "min_elev_m", "relief_m", "mean_slope_deg",
         "drain_density", "rain_annual_mean_mm",
         "temp_max_C", "temp_min_C", "temp_mean_max_C", "temp_mean_min_C"]
if INCLUDE_LATLON:
    ATTRS += ["lat", "lon"]
ATTRS = [a for a in ATTRS if a in df.columns]

X = df[ATTRS].copy()
missing = X.isna().sum()
if missing.any():
    print("Missing values per attribute:")
    print(missing[missing > 0].to_string())
    keep = X.notna().all(axis=1)
    print(f"Dropping {int((~keep).sum())} sites with incomplete attributes")
    df, X = df[keep].reset_index(drop=True), X[keep].reset_index(drop=True)

# constant / near-constant attributes break standardization
nunique = X.nunique()
dead = nunique[nunique <= 1].index.tolist()
if dead:
    print(f"Dropping constant attributes: {dead}")
    X = X.drop(columns=dead); ATTRS = [a for a in ATTRS if a not in dead]

print(f"\nAttributes used ({len(ATTRS)}): {ATTRS}")


# =============================================================================
# 2. TRANSFORM + STANDARDIZE
# =============================================================================
print(); print("=" * 70); print("2. TRANSFORM + STANDARDIZE"); print("=" * 70)

skew_before = X.skew()
logged = []
for a in LOG_ATTRS:
    if a in X.columns and (X[a] > 0).all():
        X[a] = np.log10(X[a])
        logged.append(a)
X = X.rename(columns={a: f"log10_{a}" for a in logged})
ATTRS = list(X.columns)
print(f"Log10-transformed: {logged}")

cmp = pd.DataFrame({"skew_before": skew_before.values,
                    "skew_after": X.skew().values}, index=ATTRS).round(2)
print(cmp.to_string())

still_skewed = cmp[cmp.skew_after.abs() > 2]
if len(still_skewed):
    print(f"\n! Still strongly skewed (|skew|>2): {list(still_skewed.index)}")
    print("  Consider transforming these too, or check for outliers.")

Z = StandardScaler().fit_transform(X.values)


# =============================================================================
# 3. PCA
# =============================================================================
print(); print("=" * 70); print("3. PCA"); print("=" * 70)

pca_full = PCA().fit(Z)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)
n_pc = int(np.searchsorted(cum, VAR_TARGET) + 1)
kaiser = int((pca_full.explained_variance_ >= 1.0).sum())

print(f"{'PC':>4} {'eigenvalue':>11} {'var%':>7} {'cum%':>7}")
for i in range(min(len(evr), 10)):
    print(f"{i+1:>4} {pca_full.explained_variance_[i]:>11.3f} "
          f"{evr[i]*100:>6.1f}% {cum[i]*100:>6.1f}%")

print(f"\nPCs for {VAR_TARGET:.0%} variance : {n_pc}")
print(f"PCs by Kaiser (eigenvalue>1): {kaiser}")
n_pc = max(n_pc, 2)

pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
scores = pca.fit_transform(Z)

loadings = pd.DataFrame(pca.components_.T,
                        index=ATTRS,
                        columns=[f"PC{i+1}" for i in range(n_pc)]).round(3)
print("\nLoadings:")
print(loadings.to_string())
print("\nDominant attribute per PC:")
for c in loadings.columns:
    top = loadings[c].abs().sort_values(ascending=False).head(3)
    print(f"  {c}: " + ", ".join(f"{i} ({loadings.loc[i, c]:+.2f})" for i in top.index))


# =============================================================================
# 4. CLUSTERING — Ward initialise, K-means refine
# =============================================================================
print(); print("=" * 70); print("4. CLUSTERING"); print("=" * 70)

link = linkage(scores, method="ward")
contrib_mask = df["is_contributing"].to_numpy(dtype=bool)

results = []
for k in K_RANGE:
    ward_lab = fcluster(link, k, criterion="maxclust")
    init = np.array([scores[ward_lab == c].mean(axis=0) for c in np.unique(ward_lab)])
    km = KMeans(n_clusters=k, init=init, n_init=1, random_state=RANDOM_STATE).fit(scores)
    lab = km.labels_
    sil = silhouette_score(scores, lab)
    sizes = np.bincount(lab, minlength=k)
    contrib_counts = np.array([int(contrib_mask[lab == c].sum()) for c in range(k)])
    ok = bool((contrib_counts >= MIN_CONTRIB_PER_REGION).all())
    results.append({"k": k, "silhouette": sil, "min_size": sizes.min(),
                    "min_contrib": contrib_counts.min(), "testable": ok,
                    "labels": lab, "contrib_counts": contrib_counts})
    flag = "OK " if ok else "!! "
    print(f"{flag}k={k}  silhouette={sil:.3f}  smallest region={sizes.min():3d} sites  "
          f"contributing per region={list(contrib_counts)}")

viable = [r for r in results if r["testable"]]
if viable:
    best = max(viable, key=lambda r: r["silhouette"])
    print(f"\nSelected k={best['k']} (best silhouette among testable configurations)")
else:
    best = max(results, key=lambda r: r["min_contrib"])
    print(f"\n! No k gives every region >= {MIN_CONTRIB_PER_REGION} contributing sites.")
    print(f"  Falling back to k={best['k']}. Options: lower k, lower MIN_YEARS to 10,")
    print(f"  or merge the thin regions manually in Stage 3.")

labels = best["labels"]
K = best["k"]
df["region"] = labels + 1
for i in range(n_pc):
    df[f"PC{i+1}"] = scores[:, i]


# =============================================================================
# 5. REGION PROFILES
# =============================================================================
print(); print("=" * 70); print("5. REGION PROFILES"); print("=" * 70)

raw_attrs = [a for a in ["area_km2", "mean_elev_m", "mean_slope_deg", "drain_density",
                         "rain_max_mm", "temp_max_C", "lat", "lon"] if a in df.columns]
prof = df.groupby("region")[raw_attrs].median().round(2)
prof.insert(0, "n_contrib", df.groupby("region")["is_contributing"].sum().astype(int))
prof.insert(0, "n_sites", df.groupby("region").size())
print(prof.to_string())


# =============================================================================
# 6. FIGURES
# =============================================================================
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].bar(range(1, len(evr) + 1), evr * 100, color="steelblue")
ax[0, 0].plot(range(1, len(evr) + 1), cum * 100, "ro-", lw=1.5, ms=4)
ax[0, 0].axhline(VAR_TARGET * 100, ls="--", c="grey")
ax[0, 0].axvline(n_pc + 0.5, ls=":", c="red")
ax[0, 0].set_xlabel("Principal component"); ax[0, 0].set_ylabel("Variance (%)")
ax[0, 0].set_title(f"Scree — {n_pc} PCs retained ({cum[n_pc-1]:.1%})")

dendrogram(link, truncate_mode="lastp", p=25, ax=ax[0, 1],
           color_threshold=link[-(K-1), 2], no_labels=True)
ax[0, 1].set_title(f"Ward dendrogram (cut at k={K})")

ks = [r["k"] for r in results]; sils = [r["silhouette"] for r in results]
cols = ["seagreen" if r["testable"] else "lightcoral" for r in results]
ax[1, 0].bar(ks, sils, color=cols)
ax[1, 0].axvline(K, ls=":", c="red")
ax[1, 0].set_xlabel("k"); ax[1, 0].set_ylabel("Silhouette")
ax[1, 0].set_title("Cluster quality (red = region too thin to test)")

sc = ax[1, 1].scatter(df["lon"], df["lat"], c=df["region"], cmap="tab10", s=18)
cm = df[df["is_contributing"]]
ax[1, 1].scatter(cm["lon"], cm["lat"], facecolors="none", edgecolors="k", s=55, lw=0.7)
ax[1, 1].set_xlabel("Longitude"); ax[1, 1].set_ylabel("Latitude")
ax[1, 1].set_title("Regions in space (circled = contributing)")
plt.colorbar(sc, ax=ax[1, 1], label="region")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=130)
print(f"\nSaved figure: {OUT_FIG}")


# =============================================================================
# 7. SAVE
# =============================================================================
df.to_csv(OUT_REGIONS, index=False)

with open(OUT_REPORT, "w") as fh:
    fh.write(f"RFA PCA + CLUSTERING REPORT\n{'='*60}\n\n")
    fh.write(f"Sites: {len(df)}   Contributing: {int(df['is_contributing'].sum())}\n")
    fh.write(f"Attributes ({len(ATTRS)}): {ATTRS}\n")
    fh.write(f"Log-transformed: {logged}\n")
    fh.write(f"Lat/lon included: {INCLUDE_LATLON}\n\n")
    fh.write(f"PCs retained: {n_pc} ({cum[n_pc-1]:.1%} variance)\n")
    fh.write(f"Kaiser criterion: {kaiser}\n\n")
    fh.write("LOADINGS\n"); fh.write(loadings.to_string()); fh.write("\n\n")
    fh.write(f"Clusters: k={K}, silhouette={best['silhouette']:.3f}\n\n")
    fh.write("REGION PROFILES\n"); fh.write(prof.to_string()); fh.write("\n")

print(f"Saved regions: {OUT_REGIONS}")
print(f"Saved report : {OUT_REPORT}")
print(f"\nNext: Stage 3 — within-region discordancy + H1/H2/H3 heterogeneity, "
      f"then adjust regions and re-test.")