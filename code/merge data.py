"""
Merge per-station rainfall & temperature extremes into your morphometry CSV.

Assumes:
  - rainfall_data/ has ONE file per station, columns:
        station, year, max_rain, mean_rain, total_rainfall_mm
  - temp data/ has ONE file per station, columns:
        station, year, max_temp_C, max_temp_doy, max_temp_date,
        min_temp_C, min_temp_doy, min_temp_date, mean_maxT_C, mean_minT_C

For each station file, computes:
  - rainfall_max_mm  = the single highest 'max_rain' value across all years
                       (i.e. the most extreme rainfall day ever recorded)
  - temp_max_C       = the single highest 'max_temp_C' value across all years
                       (hottest day ever recorded)
  - temp_min_C       = the single LOWEST 'min_temp_C' value across all years
                       (coldest day ever recorded)

Then merges these onto your existing All_Stations_morphometry_success.csv
by station name (case/whitespace-insensitive match), and writes a new file:
    All_Stations_morphometry_with_climate.csv

EDIT THE CONFIG BLOCK BELOW, THEN RUN.
"""

import os
import glob
import re
import pandas as pd

# =============================================================================
# CONFIG — EDIT THESE
# =============================================================================
MORPHOMETRY_CSV = r"C:\Users\bsnre\Downloads\chingka kalai\All_Stations_morphometry_success.csv"
RAINFALL_FOLDER = r"C:\Users\bsnre\Downloads\chingka kalai\rainfall_data"
TEMP_FOLDER     = r"C:\Users\bsnre\Downloads\chingka kalai\temp data"
OUTPUT_CSV      = r"C:\Users\bsnre\Downloads\chingka kalai\All_Stations_morphometry_with_climate.csv"


def normalize(name):
    """Lowercase, strip, collapse whitespace/punctuation for fuzzy station-name matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df, candidates):
    """Case/whitespace-insensitive column lookup. candidates = list of likely names."""
    lookup = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in lookup:
            return lookup[key]
    return None


def read_any(path):
    """Read a station file whether it's .xlsx or .csv."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def station_name_from_file(path, df, suffix_to_strip):
    """Prefer the 'station' column inside the file; fall back to filename."""
    if "station" in df.columns and df["station"].notna().any():
        return str(df["station"].dropna().iloc[0])
    stem = os.path.splitext(os.path.basename(path))[0]
    for suf in suffix_to_strip:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
    return stem


# =============================================================================
# 1. RAINFALL — one row per station: rainfall_max_mm
# =============================================================================
rainfall_records = []
rainfall_files = glob.glob(os.path.join(RAINFALL_FOLDER, "*.xlsx")) + \
                 glob.glob(os.path.join(RAINFALL_FOLDER, "*.csv"))
print(f"Found {len(rainfall_files)} rainfall files")

for f in rainfall_files:
    try:
        df = read_any(f)
        df.columns = [str(c).strip() for c in df.columns]
        station = station_name_from_file(f, df, ["_rainfall", "rainfall"])
        max_rain_col = find_column(df, ["max_rain", "max_rainfall", "max_rainfall_mm", "maxrain"])
        if max_rain_col is None:
            print(f"  ⚠ {os.path.basename(f)}: no max-rainfall column found. "
                  f"Columns present: {list(df.columns)}")
            continue
        rainfall_max_mm = pd.to_numeric(df[max_rain_col], errors="coerce").max()
        rainfall_records.append({"station_raw": station, "rainfall_max_mm": rainfall_max_mm})
    except Exception as e:
        print(f"  ⚠ Failed to read {os.path.basename(f)}: {e}")

rainfall_df = pd.DataFrame(rainfall_records, columns=["station_raw", "rainfall_max_mm"])
if len(rainfall_df) == 0:
    print("\n⚠ WARNING: NO rainfall files were successfully read. "
          "rainfall_max_mm will be blank for every station. "
          "Check the 'Columns present' messages above to see the real header names, "
          "then add them to the max_rain_col candidate list in the script.")
rainfall_df["station_key"] = rainfall_df["station_raw"].apply(normalize)
print(f"Extracted rainfall_max_mm for {len(rainfall_df)} stations\n")


# =============================================================================
# 2. TEMPERATURE — one row per station: temp_max_C, temp_min_C
# =============================================================================
temp_records = []
temp_files = glob.glob(os.path.join(TEMP_FOLDER, "*.xlsx")) + \
             glob.glob(os.path.join(TEMP_FOLDER, "*.csv"))
print(f"Found {len(temp_files)} temperature files")

for f in temp_files:
    try:
        df = read_any(f)
        df.columns = [str(c).strip() for c in df.columns]
        station = station_name_from_file(f, df, ["_temp", "temp", "_temperature"])
        max_temp_col = find_column(df, ["max_temp_C", "max_temp", "maxtempC"])
        min_temp_col = find_column(df, ["min_temp_C", "min_temp", "mintempC"])
        if max_temp_col is None or min_temp_col is None:
            print(f"  ⚠ {os.path.basename(f)}: missing temp columns. "
                  f"Columns present: {list(df.columns)}")
            continue
        temp_max_C = pd.to_numeric(df[max_temp_col], errors="coerce").max()
        temp_min_C = pd.to_numeric(df[min_temp_col], errors="coerce").min()
        temp_records.append({
            "station_raw": station, "temp_max_C": temp_max_C, "temp_min_C": temp_min_C
        })
    except Exception as e:
        print(f"  ⚠ Failed to read {os.path.basename(f)}: {e}")

temp_df = pd.DataFrame(temp_records, columns=["station_raw", "temp_max_C", "temp_min_C"])
if len(temp_df) == 0:
    print("\n⚠ WARNING: NO temperature files were successfully read. "
          "temp_max_C / temp_min_C will be blank for every station. "
          "Check the 'Columns present' messages above.")
temp_df["station_key"] = temp_df["station_raw"].apply(normalize)
print(f"Extracted temp_max_C/temp_min_C for {len(temp_df)} stations\n")


# =============================================================================
# 3. MERGE ONTO MORPHOMETRY CSV
# =============================================================================
morph = pd.read_csv(MORPHOMETRY_CSV)
if "status" in morph.columns:
    morph = morph[morph["status"] == "SUCCESS"].reset_index(drop=True)
morph["station_key"] = morph["station"].apply(normalize)

merged = morph.merge(
    rainfall_df[["station_key", "rainfall_max_mm"]], on="station_key", how="left"
).merge(
    temp_df[["station_key", "temp_max_C", "temp_min_C"]], on="station_key", how="left"
)

n_rainfall_matched = merged["rainfall_max_mm"].notna().sum()
n_temp_matched = merged["temp_max_C"].notna().sum()
print(f"Rainfall matched: {n_rainfall_matched} / {len(merged)} stations")
print(f"Temperature matched: {n_temp_matched} / {len(merged)} stations")

unmatched_rainfall = merged[merged["rainfall_max_mm"].isna()]["station"].tolist()
unmatched_temp = merged[merged["temp_max_C"].isna()]["station"].tolist()
if unmatched_rainfall:
    print(f"\nStations with NO rainfall match ({len(unmatched_rainfall)}):")
    print(unmatched_rainfall[:20], "..." if len(unmatched_rainfall) > 20 else "")
if unmatched_temp:
    print(f"\nStations with NO temperature match ({len(unmatched_temp)}):")
    print(unmatched_temp[:20], "..." if len(unmatched_temp) > 20 else "")

merged = merged.drop(columns=["station_key"])
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved merged file: {OUTPUT_CSV}")
print(f"New columns added: rainfall_max_mm, temp_max_C, temp_min_C")