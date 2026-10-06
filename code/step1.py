"""
STEP 1 - ATTRIBUTE FINALISATION + PCA
=====================================
1a. Morphometry (status == SUCCESS) + climate merged from rainfall_data/ and temp data/
      rainfall_max_mm = highest max_rain over all years
      temp_max_C      = highest max_temp_C over all years
      temp_min_C      = lowest  min_temp_C over all years
1b. Keep stations with > 15 years of annual-maximum STREAMFLOW record
    (counted from the streamflow files themselves) and complete attributes
1c. PCA on ALL 15 catchment attributes (z-scored), ALL components computed and saved,
    retained set = first PCs reaching >= 85 % cumulative variance

Outputs go to  <DATA_DIR>/step1_outputs/ ; step1_pc_scores_for_ROI.csv (ALL PCs by default)
and step1_annual_max_series.csv are the inputs for Step 2 (ROI).
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = r"C:\Users\bsnre\Downloads\chingka kalai"
MORPH_CSV = "All_Stations_morphometry_success.csv"
RAIN_DIR = "rainfall_data"
TEMP_DIR = "temp data"
STREAMFLOW_DIR = "streamflow data"
OUT_NAME = "step1_outputs"

MIN_YEARS = 15                 # keep stations with MORE than this many years of AMS
MIN_OBS_DAYS = 0               # 0 = every year counts (prof's rule). e.g. 292 (= 80 % of a year)
                               # drops partial years whose maximum may have missed the flood peak
EXCLUDE_NAME_CONTAINS = ["duplicate"]   # stations whose name contains these words are removed
QC_JUMP_RATIO = 10             # flag a year whose max is > 10x the station's median annual max
VAR_TARGET = 0.85             # only used to REPORT the 85 % mark
PCS_FOR_ROI = "all"           # "all" = every PC is used as the attribute space downstream
                              # "retained" = only PCs up to VAR_TARGET

# All catchment attributes.  Transform: None | "log"
# Not used (GIS-processing fields, not catchment properties):
#   snap_lon, snap_lat, snap_dist_km, final_buffer, error, file, n_years
# Not used (exact algebraic duplicates of other attributes already in the set -
#   keeping both sides of an identity just triple-weights one signal in the PCA):
#   stream_len_km  = drain_density * area_km2        (max error 7e-5, i.e. exact)
#   min_elev_m, max_elev_m = mean_elev_m +/- relief/2-ish, and max_elev_m - min_elev_m = relief_m EXACTLY
#   centroid_lat/lon       = lat/lon to r = 0.998 / 0.994 (station outlet vs. catchment centroid)
ATTRS = {
    "area_km2": "log",
    "drain_density": None,
    "mean_elev_m": None,
    "relief_m": None,
    "mean_slope_deg": None,
    "lat": None,
    "lon": None,
    "rainfall_max_mm": None,
    "temp_max_C": None,
    "temp_min_C": None,
}


# =============================================================================
# HELPERS
# =============================================================================
def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def find_col(df, candidates):
    look = {normalize(c): c for c in df.columns}
    for c in candidates:
        if normalize(c) in look:
            return look[normalize(c)]
    return None


def station_from_file(path, df, suffixes):
    if "station" in df.columns and df["station"].notna().any():
        return str(df["station"].dropna().iloc[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    for s in suffixes:
        if stem.lower().endswith(s):
            stem = stem[: -len(s)]
    return stem


def list_files(folder):
    return sorted(f for ext in ("*.csv", "*.xlsx", "*.xls") for f in glob.glob(os.path.join(folder, ext)))


# =============================================================================
# 1a. MORPHOMETRY + CLIMATE
# =============================================================================
def climate_table(folder, suffixes, spec, label):
    recs = []
    for f in list_files(folder):
        try:
            df = read_table(f)
            df.columns = [str(c).strip() for c in df.columns]
            rec = {"station_key": normalize(station_from_file(f, df, suffixes))}
            for out, (cands, how) in spec.items():
                c = find_col(df, cands)
                if c is None:
                    raise KeyError(f"none of {cands} in {list(df.columns)}")
                v = pd.to_numeric(df[c], errors="coerce")
                rec[out] = v.max() if how == "max" else v.min()
            recs.append(rec)
        except Exception as e:
            print(f"   [{label}] skipped {os.path.basename(f)}: {e}")
    t = pd.DataFrame(recs, columns=["station_key"] + list(spec))
    print(f"   {label}: {len(t)} station files read")
    return t.groupby("station_key", as_index=False).agg(
        {k: ("max" if h == "max" else "min") for k, (_, h) in spec.items()})


def build_attributes(log):
    morph = pd.read_csv(os.path.join(DATA_DIR, MORPH_CSV))
    log.append(("rows in morphometry file", len(morph)))
    morph["excel_row"] = morph.index + 2
    if "status" in morph.columns:
        morph = morph[morph["status"].astype(str).str.strip().str.upper() == "SUCCESS"]
    morph = morph.reset_index(drop=True)
    log.append(("status == SUCCESS", len(morph)))
    morph = morph.drop(columns=[c for c in ("rainfall_max_mm", "temp_max_C", "temp_min_C") if c in morph])
    morph["station_key"] = morph["station"].apply(normalize)

    rain = climate_table(os.path.join(DATA_DIR, RAIN_DIR), ["_rainfall", "rainfall"],
                         {"rainfall_max_mm": (["max_rain", "max_rainfall", "max_rainfall_mm"], "max")},
                         "rainfall")
    temp = climate_table(os.path.join(DATA_DIR, TEMP_DIR), ["_temp", "temp", "_temperature"],
                         {"temp_max_C": (["max_temp_C", "max_temp"], "max"),
                          "temp_min_C": (["min_temp_C", "min_temp"], "min")},
                         "temperature")
    attr = morph.merge(rain, on="station_key", how="left").merge(temp, on="station_key", how="left")
    for col in ("rainfall_max_mm", "temp_max_C"):
        miss = attr.loc[attr[col].isna(), "station"].tolist()
        if miss:
            print(f"   no {col.split('_')[0]} data for {len(miss)}: {miss[:15]}{' ...' if len(miss) > 15 else ''}")
    return attr


# =============================================================================
# 1b. STREAMFLOW RECORD LENGTH (annual maxima)
# =============================================================================
def load_streamflow():
    """Returns (dict file_stem -> pd.Series(annual max, index=year), QC table)."""
    files = list_files(os.path.join(DATA_DIR, STREAMFLOW_DIR))
    out, qc, shown = {}, [], False
    for f in files:
        df = read_table(f)
        df.columns = [str(c).strip() for c in df.columns]
        ycol = find_col(df, ["year", "yr", "water_year"])
        qcol = next((c for c in df.columns if "max" in c.lower() and
                     any(k in c.lower() for k in ("discharge", "flow", "cumec", "q"))
                     and not any(k in c.lower() for k in ("date", "doy", "day"))), None)
        ncol = find_col(df, ["n_obs", "nobs", "n_days"])
        if ycol is None or qcol is None:
            print(f"   [streamflow] skipped {os.path.basename(f)}: columns {list(df.columns)}")
            continue
        if not shown:
            print(f"   streamflow columns used: year = '{ycol}', annual max = '{qcol}', "
                  f"days of data = '{ncol}'" + (f" (years with < {MIN_OBS_DAYS} days dropped)" if MIN_OBS_DAYS else ""))
            shown = True
        keep = [ycol, qcol] + ([ncol] if ncol else [])
        d = df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=[ycol, qcol])
        d = d[d[qcol] > 0]
        n_all = d[ycol].nunique()
        n_partial = int((d[ncol] < 292).sum()) if ncol else np.nan
        if ncol and MIN_OBS_DAYS:
            d = d[d[ncol] >= MIN_OBS_DAYS]
        s = d.groupby(ycol)[qcol].max()
        stem = os.path.splitext(os.path.basename(f))[0]
        out[stem] = s
        med = s.median() if len(s) else np.nan
        jumps = s[s > QC_JUMP_RATIO * med] if len(s) else s
        tiny = s[s < med / QC_JUMP_RATIO] if len(s) else s
        qc.append(dict(streamflow_file=stem, years_in_file=n_all, years_used=len(s),
                       partial_years_lt_292_days=n_partial, median_annual_max=med,
                       max_annual_max=s.max() if len(s) else np.nan,
                       max_over_median=s.max() / med if len(s) else np.nan,
                       years_gt_10x_median=";".join(f"{int(y)}:{v:g}" for y, v in jumps.items()),
                       years_lt_median_over_10=";".join(f"{int(y)}:{v:g}" for y, v in tiny.items())))
    return out, pd.DataFrame(qc)


def match_streamflow(attr, sf):
    """1st choice: the 'file' column of the morphometry table.
    Fallback: station name = file stem, else station name as a whole word inside the stem."""
    stems = {k: k.lower() for k in sf}
    series, where, how = {}, {}, {}
    for idx, r in attr.iterrows():
        hit = []
        if "file" in attr.columns and pd.notna(r["file"]) and str(r["file"]).strip():
            fstem = os.path.splitext(os.path.basename(str(r["file"]).strip()))[0].lower()
            hit = [k for k, v in stems.items() if v == fstem]
            if hit:
                how[idx] = "file column"
        if not hit:
            key = str(r["station_name"]).strip().lower()
            hit = [k for k, v in stems.items() if v == key] or sorted(
                [k for k, v in stems.items()
                 if re.search(rf"(^|[^0-9a-z]){re.escape(key)}([^0-9a-z]|$)", v)], key=len)
            if hit:
                how[idx] = "station name"
        if hit:
            series[idx], where[idx] = sf[hit[0]], hit[0]
    return series, where, how


def duplicate_report(attr, out):
    """Flag repeated station names, repeated streamflow files, identical attribute rows."""
    cols = list(ATTRS)
    rows = []
    for col, label in (("station_name", "same station name"), ("streamflow_file", "same streamflow file")):
        d = attr[attr[col].notna() & attr[col].duplicated(keep=False)]
        for v, g in d.groupby(col):
            rows.append(dict(issue=label, value=v, rows=";".join(map(str, g["excel_row"])),
                             stations=";".join(g["station"].astype(str))))
    key = attr[cols].round(4).apply(lambda r: "|".join(map(str, r.values)), axis=1)
    d = attr[key.duplicated(keep=False) & attr[cols].notna().all(axis=1)]
    for _, g in d.groupby(key[d.index]):
        rows.append(dict(issue="identical attributes", value="",
                         rows=";".join(map(str, g["excel_row"])), stations=";".join(g["station"].astype(str))))
    rep = pd.DataFrame(rows, columns=["issue", "value", "rows", "stations"])
    rep.to_csv(f"{out}/step1_duplicate_check.csv", index=False)
    return rep


# =============================================================================
# 1c. PCA
# =============================================================================
def run_pca(attr, out):
    cols = list(ATTRS)
    X = attr[cols].astype(float).copy()
    for c, tr in ATTRS.items():
        if tr == "log":
            X[c] = np.log(X[c].clip(lower=1e-6))

    summ = attr[cols].describe().T
    summ["skew_raw"] = attr[cols].skew()
    summ["transform"] = [ATTRS[c] or "" for c in cols]
    summ["skew_used"] = X.skew()
    summ.to_csv(f"{out}/step1_attribute_summary.csv")

    corr = X.corr()
    corr.to_csv(f"{out}/step1_attribute_correlation.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax); ax.set_title("Attribute correlation (after transforms)")
    fig.tight_layout(); fig.savefig(f"{out}/step1_correlation_heatmap.png", dpi=200); plt.close(fig)

    Xz = StandardScaler().fit_transform(X.values)
    pca = PCA().fit(Xz)                                   # ALL components
    ev, evr = pca.explained_variance_, pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    npc = int(np.searchsorted(cum, VAR_TARGET) + 1)
    pcs = [f"PC{i + 1}" for i in range(len(ev))]

    var = pd.DataFrame({"PC": pcs, "eigenvalue": ev, "variance_%": evr * 100,
                        "cumulative_%": cum * 100, "kaiser_eig>1": ev > 1,
                        "within_85pct": [i < npc for i in range(len(ev))],
                        "used_for_ROI": [PCS_FOR_ROI == "all" or i < npc for i in range(len(ev))]})
    var.to_csv(f"{out}/step1_pca_variance_all_components.csv", index=False)

    pd.DataFrame(pca.components_.T, index=cols, columns=pcs).to_csv(
        f"{out}/step1_pca_eigenvectors_all_components.csv")
    load = pd.DataFrame(pca.components_.T * np.sqrt(ev), index=cols, columns=pcs)
    load.to_csv(f"{out}/step1_pca_loadings_all_components.csv")
    comm = (load.iloc[:, :npc] ** 2).sum(axis=1).rename(f"communality_first_{npc}_PCs")
    comm.to_csv(f"{out}/step1_communalities.csv")

    scores = pca.transform(Xz)
    sc = pd.DataFrame(scores, columns=pcs); sc.insert(0, "station", attr["station"].values)
    sc.to_csv(f"{out}/step1_pc_scores_all_components.csv", index=False)
    n_use = len(pcs) if PCS_FOR_ROI == "all" else npc
    sc.iloc[:, : n_use + 1].to_csv(f"{out}/step1_pc_scores_for_ROI.csv", index=False)

    # scree
    x = np.arange(1, len(ev) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar(x, evr * 100, color=["#4c72b0" if i < npc else "#b0b0b0" for i in range(len(ev))])
    ax[0].plot(x, cum * 100, "o-", c="#dd8452")
    ax[0].axhline(VAR_TARGET * 100, ls="--", c="grey")
    ax[0].set_xticks(x); ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Variance (%)")
    ax[0].set_title(f"Scree: {VAR_TARGET:.0%} reached at PC{npc} ({cum[npc - 1] * 100:.1f} %)")
    ax[1].plot(x, ev, "o-"); ax[1].axhline(1, ls="--", c="grey")
    ax[1].set_xticks(x); ax[1].set_xlabel("Principal component"); ax[1].set_ylabel("Eigenvalue")
    ax[1].set_title("Eigenvalues (Kaiser line = 1)")
    fig.tight_layout(); fig.savefig(f"{out}/step1_scree.png", dpi=200); plt.close(fig)

    # loadings heatmap, all components, retained ones boxed
    fig, ax = plt.subplots(figsize=(0.7 * len(pcs) + 3, 6.5))
    im = ax.imshow(load.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pcs))); ax.set_xticklabels(pcs, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(pcs)):
            ax.text(j, i, f"{load.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.axvline(npc - 0.5, c="k", lw=2)
    fig.colorbar(im, ax=ax, label="loading (correlation)")
    ax.set_title(f"PCA loadings - all components (black line = {VAR_TARGET:.0%} variance mark)")
    fig.tight_layout(); fig.savefig(f"{out}/step1_loadings_all_components.png", dpi=200); plt.close(fig)

    # biplot PC1-PC2
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(scores[:, 0], scores[:, 1], s=14, c="#4c72b0", alpha=0.7)
    f = np.abs(scores[:, :2]).max() * 0.9
    for c in cols:
        ax.arrow(0, 0, load.loc[c, "PC1"] * f, load.loc[c, "PC2"] * f, color="#c44e52", head_width=0.05 * f / 3)
        ax.text(load.loc[c, "PC1"] * f * 1.1, load.loc[c, "PC2"] * f * 1.1, c, fontsize=7, color="#c44e52")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f} %)"); ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f} %)")
    ax.axhline(0, c="grey", lw=0.5); ax.axvline(0, c="grey", lw=0.5); ax.set_title("Biplot PC1-PC2")
    fig.tight_layout(); fig.savefig(f"{out}/step1_biplot_PC1_PC2.png", dpi=200); plt.close(fig)

    hi = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(cols) for b in cols[i + 1:]
          if abs(corr.loc[a, b]) > 0.9]
    return var, load, npc, hi


# =============================================================================
def main():
    out = os.path.join(DATA_DIR, OUT_NAME)
    os.makedirs(out, exist_ok=True)
    log = []

    print("1a. Morphometry + climate")
    attr = build_attributes(log)

    excl = attr["station"].astype(str).str.lower().str.contains(
        "|".join(EXCLUDE_NAME_CONTAINS), regex=True) if EXCLUDE_NAME_CONTAINS else pd.Series(False, index=attr.index)

    # unique id: repeated names get a suffix with their Excel row number
    attr["station_name"] = attr["station"].astype(str)
    dupn = attr["station_name"].duplicated(keep=False)
    attr["station"] = np.where(dupn, attr["station_name"] + "__row" + attr["excel_row"].astype(str),
                               attr["station_name"])

    print("1b. Streamflow record length")
    sf, qc = load_streamflow()
    series, where, how = match_streamflow(attr, sf)
    attr["streamflow_file"] = pd.Series(where)
    attr["matched_by"] = pd.Series(how)
    attr["n_ams_years"] = [len(series[i]) if i in series else 0 for i in attr.index]
    dup = duplicate_report(attr, out)

    reasons = pd.Series("kept", index=attr.index)
    reasons[excl] = "name marked " + "/".join(EXCLUDE_NAME_CONTAINS)
    reasons[(reasons == "kept") & attr["streamflow_file"].isna()] = "no streamflow file matched"
    reasons[(reasons == "kept") & (attr["n_ams_years"] <= MIN_YEARS)] = f"<= {MIN_YEARS} years of AMS"
    miss = attr[list(ATTRS)].isna()
    reasons[(reasons == "kept") & miss.any(axis=1)] = "missing attribute(s): " + \
        miss.apply(lambda r: ",".join(r.index[r]), axis=1)
    attr["step1_status"] = reasons
    if not qc.empty:
        qc = attr[["station", "streamflow_file", "step1_status"]].merge(qc, on="streamflow_file", how="right")
        qc["flag"] = np.where((qc.years_gt_10x_median != "") | (qc.years_lt_median_over_10 != ""),
                              "CHECK: possible unit error / spike / near-zero year", "")
        qc.to_csv(f"{out}/step1_streamflow_qc.csv", index=False)
    attr.drop(columns="station_key").to_csv(f"{out}/step1_all_stations_with_status.csv", index=False)

    log.append(("streamflow file matched", int(attr["streamflow_file"].notna().sum())))
    log.append(("   of which via 'file' column", int((attr["matched_by"] == "file column").sum())))
    log.append((f"> {MIN_YEARS} years AMS", int((attr["n_ams_years"] > MIN_YEARS).sum())))
    keep_idx = attr.index[reasons == "kept"]
    final = attr.loc[keep_idx].reset_index(drop=True)
    log.append(("complete attributes -> FINAL", len(final)))

    final.drop(columns="station_key").to_csv(f"{out}/step1_final_attributes.csv", index=False)
    ams = pd.concat([series[i].rename("ams").rename_axis("year").reset_index()
                     .assign(station=attr.at[i, "station"]) for i in keep_idx],
                    ignore_index=True)[["station", "year", "ams"]]
    ams.to_csv(f"{out}/step1_annual_max_series.csv", index=False)

    print("1c. PCA")
    var, load, npc, hi = run_pca(final, out)

    print("\n================ STATION COUNT ================")
    for k, v in log:
        print(f"  {k:<32} {v}")
    dr = reasons[reasons != "kept"].str.split(":").str[0].value_counts()
    print("  drop reasons: none" if dr.empty else
          "  drop reasons:\n    " + dr.to_string().replace("\n", "\n    "))
    print("\n================ PCA (all components) ================")
    print(var.round(3).to_string(index=False))
    print(f"\n  85 % variance reached at PC{npc} ({var['cumulative_%'].iloc[npc - 1]:.1f} %)")
    print(f"  PCs passed to Step 2 (ROI): {int(var['used_for_ROI'].sum())} "
          f"({'ALL components' if PCS_FOR_ROI == 'all' else 'retained only'})")
    print("\n  Loadings (all components):")
    print(load.round(2).to_string())
    if hi:
        print("\n  Highly correlated attribute pairs (|r| > 0.9):")
        for a, b, r in hi:
            print(f"    {a:<16} {b:<16} {r}")
    if not qc.empty:
        fl = qc[(qc.flag != "") & (qc.step1_status == "kept")]
        print(f"\n  STREAMFLOW QC (step1_streamflow_qc.csv): {len(fl)} kept stations to check")
        if len(fl):
            print("    " + fl[["station", "median_annual_max", "max_annual_max", "years_gt_10x_median",
                            "years_lt_median_over_10"]].head(25).round(1).to_string(index=False)
                  .replace("\n", "\n    "))
        if qc.partial_years_lt_292_days.notna().any():
            k = qc[qc.step1_status == "kept"]
            print(f"    partial years (< 292 days) in kept stations: "
                  f"{int(k.partial_years_lt_292_days.sum())} of {int(k.years_in_file.sum())} station-years")
    if not dup.empty:
        print("\n  DUPLICATES (see step1_duplicate_check.csv; Excel row numbers):")
        print("    " + dup.to_string(index=False).replace("\n", "\n    "))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()