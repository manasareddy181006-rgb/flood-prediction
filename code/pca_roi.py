"""
PCA -> K-means clustering -> Region-of-Influence (ROI) delineation
for your 282-station regionalization project.

Pipeline:
    1. Load attribute table (one row per station)
    2. Standardize attributes (z-score)
    3. Run PCA -> scree plot -> pick components explaining ~80-90% variance
    4. K-means clustering on the retained PC scores
       - elbow method + silhouette score to help choose K
    5. Region-of-Influence (ROI): for EACH station, build its own region as
       the set of stations within a chosen distance threshold in PC space,
       weighted so nearer stations count more (Burn 1990 style)
    6. Save: PCA loadings, scree plot, cluster assignments, cluster map,
       ROI membership lists, ROI region-size (station-years) summary

EDIT THE CONFIG BLOCK BELOW FIRST.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# =============================================================================
# CONFIG — EDIT THESE TO MATCH YOUR ACTUAL CSV
# =============================================================================
CSV_PATH = r"C:\Users\bsnre\Downloads\chingka kalai\All_Stations_morphometry_success.csv"
OUTPUT_DIR = r"C:\Users\bsnre\Downloads\chingka kalai\Output_PCA_Clustering"

STATION_ID_COL = "station"

# Confirmed from your actual CSV. rainfall_max / streamflow / temp_min / temp_max
# are NOT in this file yet — they live in your separate rainfall_data,
# streamflow data, and temp data folders. Set to None here until you've merged
# them in (see note below the CONFIG block).
COLUMN_MAP = {
    "slope":        "mean_slope_deg",
    "area":         "area_km2",
    "lat":          "lat",
    "lon":          "lon",
    "rainfall_max": None,   # <-- not in this CSV yet; merge in once extracted
    "streamflow":   None,   # <-- not in this CSV yet; merge in once extracted
    "temp_min":     None,   # <-- not in this CSV yet; merge in once extracted
    "temp_max":     None,   # <-- not in this CSV yet; merge in once extracted
}

RECORD_LENGTH_COL = "n_years"   # <-- this is your actual record-length column

MIN_RECORD_YEARS = 15   # <-- only stations with > this many years of
                         #     streamflow record are kept for PCA/clustering

N_PCS = None            # None = auto-pick via 85% variance threshold; or set an int
K_RANGE = range(2, 13)  # candidate K values to test for K-means
ROI_THRESHOLD = 1.5     # distance threshold in standardized PC space for ROI
                         # (start here, tune based on the ROI size plot this
                         # script produces)
MIN_STATION_YEARS = 5 * 100  # 5T rule of thumb (Reed et al. 1999); edit T as needed

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} stations, columns: {list(df.columns)}")

if "status" in df.columns:
    n_before_status = len(df)
    df = df[df["status"] == "SUCCESS"].reset_index(drop=True)
    print(f"Filtered to status == SUCCESS: {len(df)} / {n_before_status} stations kept")

attr_cols = [v for v in COLUMN_MAP.values() if v is not None]
required_cols = attr_cols + [STATION_ID_COL]
if RECORD_LENGTH_COL:
    required_cols = required_cols + [RECORD_LENGTH_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"These columns are not in your CSV: {missing}\n"
        f"Available columns are: {list(df.columns)}\n"
        f"Fix COLUMN_MAP / STATION_ID_COL / RECORD_LENGTH_COL at the top of the script."
    )

# ── Filter: only stations with > MIN_RECORD_YEARS of streamflow record ──
n_before = len(df)
if RECORD_LENGTH_COL:
    df = df[df[RECORD_LENGTH_COL] > MIN_RECORD_YEARS].reset_index(drop=True)
    print(f"Filtered to stations with > {MIN_RECORD_YEARS} years of record: "
          f"{len(df)} / {n_before} stations kept")
else:
    print("WARNING: RECORD_LENGTH_COL is None — no record-length filter applied. "
          "Set it if you want the >15-year filter enforced.")

work = df[[STATION_ID_COL] + attr_cols].dropna().reset_index(drop=True)
print(f"After dropping rows with missing attributes: {len(work)} stations")

X_raw = work[attr_cols].values


# =============================================================================
# 2. STANDARDIZE
# =============================================================================
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)


# =============================================================================
# 3. PCA
# =============================================================================
pca_full = PCA()
scores_full = pca_full.fit_transform(X)
explained = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained)

if N_PCS is None:
    n_pcs = int(np.argmax(cum_explained >= 0.85) + 1)
    n_pcs = max(n_pcs, 2)
else:
    n_pcs = N_PCS
print(f"Retaining {n_pcs} PCs, explaining {cum_explained[n_pcs-1]*100:.1f}% of variance")

pca = PCA(n_components=n_pcs)
pc_scores = pca.fit_transform(X)

# Loadings: how much each original attribute contributes to each PC
loadings = pd.DataFrame(
    pca.components_.T,
    index=attr_cols,
    columns=[f"PC{i+1}" for i in range(n_pcs)]
)
loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))
print("\nPCA loadings:\n", loadings.round(3))

# ── Scree plot ───────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
n_show = min(len(explained), 10)
ax1.bar(range(1, n_show + 1), explained[:n_show] * 100, color="#1a6eb5", alpha=0.75, label="Individual")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Variance explained (%)")
ax2 = ax1.twinx()
ax2.plot(range(1, n_show + 1), cum_explained[:n_show] * 100, color="#e74c3c",
          marker="o", label="Cumulative")
ax2.axhline(85, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("Cumulative variance (%)")
ax1.set_title(f"Scree Plot — retained {n_pcs} PCs ({cum_explained[n_pcs-1]*100:.1f}% variance)")
fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scree_plot.png"), dpi=150)
plt.close()


# =============================================================================
# 4. K-MEANS — elbow + silhouette to help pick K
# =============================================================================
inertias, sil_scores = [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(pc_scores)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(pc_scores, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(K_RANGE), inertias, marker="o", color="#1a6eb5")
axes[0].set_xlabel("Number of clusters K")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow Method")

axes[1].plot(list(K_RANGE), sil_scores, marker="o", color="#e74c3c")
axes[1].set_xlabel("Number of clusters K")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Method (higher = better)")
best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
axes[1].axvline(best_k, color="gray", linestyle="--")
axes[1].annotate(f"best K={best_k}", (best_k, max(sil_scores)),
                  textcoords="offset points", xytext=(10, 0))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_k_selection.png"), dpi=150)
plt.close()
print(f"\nSuggested K (by silhouette): {best_k}")
print("Inspect kmeans_k_selection.png and pick K manually if the suggestion looks off.")

# ── Final K-means with chosen K ─────────────────────────────────────────
FINAL_K = best_k   # <-- override manually here once you've looked at the plot, e.g. FINAL_K = 6
kmeans_final = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(pc_scores)
work["kmeans_cluster"] = cluster_labels

# ── PC1 vs PC2 scatter colored by cluster ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(pc_scores[:, 0], pc_scores[:, 1], c=cluster_labels,
                      cmap="tab10", s=40, edgecolor="white", linewidth=0.4)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"K-means Clusters in PC1–PC2 Space (K={FINAL_K})")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
ax.add_artist(legend1)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters_pc_space.png"), dpi=150)
plt.close()

# ── Geographic map of clusters (if lat/lon available) ───────────────────
if COLUMN_MAP.get("lat") and COLUMN_MAP.get("lon"):
    fig, ax = plt.subplots(figsize=(9, 10))
    sc = ax.scatter(work[COLUMN_MAP["lon"]], work[COLUMN_MAP["lat"]],
                     c=cluster_labels, cmap="tab10", s=45,
                     edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Geographic Distribution of K-means Clusters (K={FINAL_K})")
    legend1 = ax.legend(*sc.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_clusters_map.png"), dpi=150)
    plt.close()

work.to_csv(os.path.join(OUTPUT_DIR, "stations_with_kmeans_clusters.csv"), index=False)


# =============================================================================
# 5. REGION OF INFLUENCE (ROI) — Burn (1990a,b) style
#
# Unlike K-means (fixed partitions), ROI gives EACH station its own region:
# all stations within a distance threshold in (standardized) PC space,
# optionally weighted by proximity.
# =============================================================================
dist_matrix = cdist(pc_scores, pc_scores, metric="euclidean")

roi_records = []
roi_sizes = []
for i in range(len(work)):
    within = np.where(dist_matrix[i] <= ROI_THRESHOLD)[0]
    members = work.loc[within, STATION_ID_COL].tolist()
    roi_sizes.append(len(members))
    if RECORD_LENGTH_COL and RECORD_LENGTH_COL in df.columns:
        years = df.loc[df[STATION_ID_COL].isin(members), RECORD_LENGTH_COL].sum()
    else:
        years = np.nan
    roi_records.append({
        "target_station": work.loc[i, STATION_ID_COL],
        "roi_size_stations": len(members),
        "roi_total_station_years": years,
        "roi_members": ";".join(map(str, members)),
    })

roi_df = pd.DataFrame(roi_records)
roi_df.to_csv(os.path.join(OUTPUT_DIR, "roi_regions.csv"), index=False)

# ── Distribution of ROI sizes (helps you tune ROI_THRESHOLD) ────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(roi_sizes, bins=20, color="#1a6eb5", edgecolor="white")
ax.set_xlabel("Number of stations in ROI region")
ax.set_ylabel("Number of target stations")
ax.set_title(f"ROI Region Size Distribution (threshold={ROI_THRESHOLD})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roi_size_distribution.png"), dpi=150)
plt.close()

if RECORD_LENGTH_COL and RECORD_LENGTH_COL in df.columns:
    n_insufficient = (roi_df["roi_total_station_years"] < MIN_STATION_YEARS).sum()
    print(f"\nROI regions with < {MIN_STATION_YEARS} station-years "
          f"(5T rule of thumb): {n_insufficient} / {len(roi_df)}")
    print("Consider raising ROI_THRESHOLD if this number is large.")

print(f"\nDone. All outputs in: {OUTPUT_DIR}")
print("Files: pca_loadings.csv, scree_plot.png, kmeans_k_selection.png,")
print("       kmeans_clusters_pc_space.png, kmeans_clusters_map.png,")
print("       stations_with_kmeans_clusters.csv, roi_regions.csv,")
print("       roi_size_distribution.png")