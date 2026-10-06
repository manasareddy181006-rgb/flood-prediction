import os
import glob
import traceback
import requests
import pandas as pd
import numpy as np
from collections import deque
from pysheds.grid import Grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd
from rasterio.features import shapes
import rasterio
from rasterio.windows import from_bounds


# ── Patch for newer NumPy versions ────────────────────────────────────────
if not hasattr(np, "in1d"):
    np.in1d = np.isin


# =============================================================================
# CONFIG — EDIT THESE PATHS
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
EXCEL_FOLDER = r"d:\Manasa_ce24b050\Chingka_lai\All Stations"
OUTPUT_ROOT  = os.path.join(BASE_DIR, "Output")
MOSAIC_PATH  = r"D:\Manasa_ce24b050\Chingka_lai\India_DEM\india_dem_cropped.tif"

# =============================================================================
# DELINEATION PARAMS
# =============================================================================
BUFFER_START     = 1.00
MAX_BUFFER       = 5.50
BUFFER_STEP      = 0.50
MIN_AREA_KM2     = 10
STREAM_THRESHOLD = 1000
SNAP_RADIUS_INIT = 0.05   # initial snap search radius (degrees, ~5km)
SNAP_RADIUS_MIN  = 0.005  # minimum snap radius (degrees, ~500m)

# =============================================================================
# OUTPUT FOLDERS
# =============================================================================
DIR_SHP   = os.path.join(OUTPUT_ROOT, "Shapefiles")
DIR_PLOTS = os.path.join(OUTPUT_ROOT, "Plots")
DIR_DEMS  = os.path.join(OUTPUT_ROOT, "DEMs")
DIR_CSV   = os.path.join(OUTPUT_ROOT, "Results")

for d in [DIR_SHP, DIR_PLOTS, DIR_DEMS, DIR_CSV]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_filename(name):
    return ("".join(c if c.isalnum() or c in "_- " else "_"
                    for c in name).strip().replace(" ", "_"))


def get_station_info(excel_path):
    try:
        meta         = pd.read_excel(excel_path, sheet_name=0, header=None)
        station_name = str(meta.iloc[9,  1]).strip()
        lat          = float(meta.iloc[21, 1])
        lon          = float(meta.iloc[22, 1])
        return station_name, lat, lon
    except Exception as e:
        print(f"    ERROR reading metadata: {e}")
        return None, None, None


def get_annual_max(excel_path):
    try:
        df = pd.read_excel(excel_path, sheet_name=1, skiprows=6)
        df.columns = df.columns.astype(str).str.strip()
        df["Date"] = pd.to_datetime(
            df["Data Time"].astype(str).str.strip(),
            format="%Y-%m-%dT%H:%M:%S",
            errors="coerce"
        )
        df["Flow"]   = pd.to_numeric(df["Data Value"], errors="coerce")
        df_clean     = df.dropna(subset=["Date", "Flow"]).copy()
        df_clean["Year"] = df_clean["Date"].dt.year
        return df_clean.groupby("Year")["Flow"].max()
    except Exception:
        return None


def clip_dem_from_mosaic(lat, lon, buffer, dem_path):
    """Clip a DEM tile from the India mosaic — no API calls needed."""
    try:
        with rasterio.open(MOSAIC_PATH) as src:
            window = from_bounds(
                left      = lon - buffer,
                bottom    = lat - buffer,
                right     = lon + buffer,
                top       = lat + buffer,
                transform = src.transform
            )
            data      = src.read(window=window)
            transform = src.window_transform(window)
            meta      = src.meta.copy()
            meta.update({
                "height"    : data.shape[1],
                "width"     : data.shape[2],
                "transform" : transform
            })
            with rasterio.open(dem_path, "w", **meta) as dst:
                dst.write(data)
        return True
    except Exception as e:
        print(f"    ⚠ Mosaic clip failed: {e}")
        return False


def catchment_touches_boundary(catch_mask):
    return (np.any(catch_mask[0, :])  or
            np.any(catch_mask[-1, :]) or
            np.any(catch_mask[:, 0])  or
            np.any(catch_mask[:, -1]))


def check_partial_overlap(new_polygon, station_name, dir_shp):
    """
    Check if a newly delineated catchment partially overlaps any existing shapefile.
    Returns (True, overlapping_station_name, overlap_pct) if bad overlap found.
    Full containment (>99%) is acceptable — that is nested catchments.
    Partial overlap (10-99%) means snapping error — needs tighter snap radius.
    """
    shp_files = glob.glob(os.path.join(dir_shp, "*.shp"))
    for shp in shp_files:
        try:
            existing = gpd.read_file(shp)
            for _, row in existing.iterrows():
                if row.get("station") == station_name:
                    continue  # skip self
                if new_polygon.intersects(row.geometry):
                    overlap = new_polygon.intersection(row.geometry).area
                    smaller = min(new_polygon.area, row.geometry.area)
                    pct = overlap / smaller * 100
                    if 10 < pct < 99:
                        return True, row.get("station", "unknown"), pct
        except Exception:
            pass
    return False, None, None


def snap_outlet(acc_array, grid, lon, lat, search_radius=0.05):
    dx = abs(grid.affine.a)
    dy = abs(grid.affine.e)
    col_out = int((lon - grid.extent[0]) / dx)
    row_out = int((grid.extent[3] - lat) / dy)
    col_out = max(0, min(acc_array.shape[1] - 1, col_out))
    row_out = max(0, min(acc_array.shape[0] - 1, row_out))
    px = max(1, int(search_radius / dx))
    r0 = max(0, row_out - px);  r1 = min(acc_array.shape[0], row_out + px)
    c0 = max(0, col_out - px);  c1 = min(acc_array.shape[1], col_out + px)
    window = acc_array[r0:r1, c0:c1]
    for threshold in [500000, 100000, 50000, 10000, 1000]:
        mask = window >= threshold
        if mask.sum() == 0:
            continue
        rows_t, cols_t = np.where(mask)
        dists  = ((rows_t - (row_out - r0)) ** 2 +
                  (cols_t - (col_out - c0)) ** 2)
        best     = np.argmin(dists)
        best_row = r0 + rows_t[best]
        best_col = c0 + cols_t[best]
        x_snap   = grid.extent[0] + (best_col + 0.5) * dx
        y_snap   = grid.extent[3] - (best_row + 0.5) * dy
        return x_snap, y_snap, threshold
    return None, None, None


def bfs_catchment(fdir_array, snap_row, snap_col):
    d8_map = {
        1:   (0,  1),
        2:   (1,  1),
        4:   (1,  0),
        8:   (1, -1),
        16:  (0, -1),
        32:  (-1, -1),
        64:  (-1,  0),
        128: (-1,  1),
    }
    rows, cols = fdir_array.shape
    catch_mask = np.zeros((rows, cols), dtype=bool)
    catch_mask[snap_row, snap_col] = True
    queue = deque()
    queue.append((snap_row, snap_col))

    while queue:
        r, c = queue.popleft()
        for dr, dc in d8_map.values():
            nr, nc = r - dr, c - dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not catch_mask[nr, nc]:
                    nd = fdir_array[nr, nc]
                    if nd in d8_map:
                        ndr, ndc = d8_map[nd]
                        if nr + ndr == r and nc + ndc == c:
                            catch_mask[nr, nc] = True
                            queue.append((nr, nc))
    return catch_mask


def make_hillshade(dem_array, azimuth=315, altitude=45):
    az  = np.radians(azimuth)
    alt = np.radians(altitude)
    dy_g, dx_g = np.gradient(np.where(np.isnan(dem_array), 0, dem_array))
    slope  = np.arctan(np.sqrt(dx_g**2 + dy_g**2))
    aspect = np.arctan2(-dy_g, dx_g)
    shaded = (np.sin(alt) * np.cos(slope) +
              np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shaded = np.clip(shaded, 0, 1)
    shaded[np.isnan(dem_array)] = np.nan
    return shaded


# =============================================================================
# MAIN DELINEATION FUNCTION
# =============================================================================

def delineate_station(station_name, lat, lon):
    safe = safe_filename(station_name)

    result = dict(
        station        = station_name,
        lat            = lat,
        lon            = lon,
        status         = "FAILED",
        area_km2       = None,
        mean_elev_m    = None,
        min_elev_m     = None,
        max_elev_m     = None,
        relief_m       = None,
        mean_slope_deg = None,
        drain_density  = None,
        stream_len_km  = None,
        centroid_lon   = None,
        centroid_lat   = None,
        snap_lon       = None,
        snap_lat       = None,
        snap_dist_km   = None,
        final_buffer   = None,
        error          = None,
    )

    buffer      = BUFFER_START
    snap_radius = SNAP_RADIUS_INIT  # starts at 0.05°, reduces if overlap detected

    while buffer <= MAX_BUFFER:

        dem_path = os.path.join(DIR_DEMS, f"{safe}_buf{buffer:.2f}.tif")

        # ── Clip DEM from mosaic (skip if already cached) ──────────────
        if not os.path.exists(dem_path):
            print(f"    Clipping DEM from mosaic buffer={buffer:.2f}°...")
            if not clip_dem_from_mosaic(lat, lon, buffer, dem_path):
                result["error"] = "Mosaic clip failed"
                return result
        else:
            print(f"    Using cached DEM buffer={buffer:.2f}°")

        # ── Condition DEM ──────────────────────────────────────────────
        try:
            grid     = Grid.from_raster(dem_path)
            dem      = grid.read_raster(dem_path)
            filled   = grid.fill_pits(dem)
            flooded  = grid.fill_depressions(filled)
            resolved = grid.resolve_flats(flooded)
            fdir     = grid.flowdir(resolved)
            acc      = grid.accumulation(fdir)
        except Exception as e:
            result["error"] = f"DEM processing error: {e}"
            return result

        acc_array  = np.array(acc)
        fdir_array = np.array(fdir).astype(np.int16)

        # ── Snap outlet with current snap radius ───────────────────────
        print(f"    Snapping with radius={snap_radius:.4f}°...")
        x_snap, y_snap, used_thr = snap_outlet(acc_array, grid, lon, lat,
                                                search_radius=snap_radius)

        if x_snap is None:
            print(f"    ⚠ No stream found — increasing buffer")
            buffer = round(buffer + BUFFER_STEP, 2)
            continue

        snap_dist_km = ((x_snap - lon)**2 + (y_snap - lat)**2)**0.5 * 111
        print(f"    Snapped | dist={snap_dist_km:.2f} km | threshold={used_thr}")

        # ── Delineate catchment — BFS ──────────────────────────────────
        try:
            dx = abs(grid.affine.a)
            dy = abs(grid.affine.e)
            snap_col = int((x_snap - grid.extent[0]) / dx)
            snap_row = int((grid.extent[3] - y_snap)  / dy)
            snap_col = max(0, min(fdir_array.shape[1] - 1, snap_col))
            snap_row = max(0, min(fdir_array.shape[0] - 1, snap_row))
            catch_mask = bfs_catchment(fdir_array, snap_row, snap_col)
        except Exception as e:
            result["error"] = f"Catchment delineation error: {e}"
            return result

        # ── Area ───────────────────────────────────────────────────────
        cell_size_m = abs(grid.affine.a) * 111000
        area_km2    = float(catch_mask.sum() * (cell_size_m ** 2) / 1e6)
        print(f"    Area = {area_km2:.1f} km²")

        if area_km2 < MIN_AREA_KM2:
            print(f"    ⚠ Area too small — increasing buffer")
            buffer = round(buffer + BUFFER_STEP, 2)
            continue

        if catchment_touches_boundary(catch_mask):
            print(f"    ✗ Catchment cut at DEM boundary — increasing buffer")
            buffer = round(buffer + BUFFER_STEP, 2)
            continue

        # ── Build polygon to check overlap BEFORE saving ───────────────
        catch_uint8 = np.ascontiguousarray(catch_mask.astype(np.uint8))
        geom_list   = [shape(g) for g, v in
                       shapes(catch_uint8, transform=grid.affine) if v == 1]

        if len(geom_list) == 0:
            print(f"    ⚠ No polygon extracted")
            buffer = round(buffer + BUFFER_STEP, 2)
            continue

        catchment_polygon = unary_union(geom_list)

        # ── Check for partial overlap with existing catchments ─────────
        has_overlap, overlap_station, overlap_pct = check_partial_overlap(
            catchment_polygon, station_name, DIR_SHP
        )

        if has_overlap:
            new_radius = round(snap_radius * 0.5, 4)
            print(f"    ⚠ Partial overlap with {overlap_station} ({overlap_pct:.1f}%)")
            if new_radius >= SNAP_RADIUS_MIN:
                print(f"    → Reducing snap radius: {snap_radius} → {new_radius}")
                snap_radius = new_radius
                continue  # retry with tighter snap, same buffer
            else:
                print(f"    ✗ Snap radius too small to reduce further — accepting overlap")

        # ── SUCCESS — compute morphometrics ────────────────────────────
        print(f"    ✓ Catchment OK | buffer={buffer:.2f}° | snap_radius={snap_radius:.4f}°")

        # Elevation stats
        full_dem_view = np.array(grid.view(dem)).astype(float)
        full_dem_view[full_dem_view <= 0] = np.nan
        catchment_dem = full_dem_view.copy()
        catchment_dem[~catch_mask] = np.nan
        valid_elev = catchment_dem[~np.isnan(catchment_dem)]

        mean_elev = float(valid_elev.mean())
        min_elev  = float(valid_elev.min())
        max_elev  = float(valid_elev.max())
        relief    = float(max_elev - min_elev)

        # Slope — crop to bounding box to save memory
        cell_deg     = abs(grid.affine.a)
        c_m_y        = cell_deg * 111000
        c_m_x        = cell_deg * 111000 * np.cos(np.radians(lat))
        rows_with_catch = np.where(catch_mask.any(axis=1))[0]
        cols_with_catch = np.where(catch_mask.any(axis=0))[0]
        r0, r1 = rows_with_catch[0], rows_with_catch[-1] + 1
        c0, c1 = cols_with_catch[0], cols_with_catch[-1] + 1
        catchment_dem_crop = catchment_dem[r0:r1, c0:c1]
        catch_mask_crop    = catch_mask[r0:r1, c0:c1]
        dem_fill           = np.where(np.isnan(catchment_dem_crop), 0, catchment_dem_crop)
        dz_dy, dz_dx       = np.gradient(dem_fill, c_m_y, c_m_x)
        slope_deg          = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        slope_deg[~catch_mask_crop] = np.nan
        mean_slope         = float(slope_deg[~np.isnan(slope_deg)].mean())

        # Drainage density
        stream_net    = (acc_array[r0:r1, c0:c1] > STREAM_THRESHOLD) & catch_mask_crop
        stream_len_km = float(stream_net.sum() * 30 / 1000)
        drain_density = stream_len_km / area_km2

        # ── Shapefile ──────────────────────────────────────────────────
        centroid_lon = round(catchment_polygon.centroid.x, 5)
        centroid_lat = round(catchment_polygon.centroid.y, 5)

        try:
            gdf = gpd.GeoDataFrame(
                {
                    "station"     : [station_name],
                    "area_km2"    : [round(area_km2,      2)],
                    "mean_elev_m" : [round(mean_elev,     2)],
                    "min_elev_m"  : [round(min_elev,      2)],
                    "max_elev_m"  : [round(max_elev,      2)],
                    "relief_m"    : [round(relief,        2)],
                    "slope_deg"   : [round(mean_slope,    2)],
                    "drain_den"   : [round(drain_density, 4)],
                    "stream_km"   : [round(stream_len_km, 2)],
                    "outlet_lon"  : [round(x_snap,        5)],
                    "outlet_lat"  : [round(y_snap,        5)],
                    "snap_km"     : [round(snap_dist_km,  3)],
                    "snap_rad"    : [round(snap_radius,   4)],
                    "centroid_lon": [centroid_lon],
                    "centroid_lat": [centroid_lat],
                    "buffer_deg"  : [buffer],
                    "gauge_lon"   : [lon],
                    "gauge_lat"   : [lat],
                },
                geometry=[catchment_polygon],
                crs="EPSG:4326"
            )
            shp_out = os.path.join(DIR_SHP, f"{safe}.shp")
            gdf.to_file(shp_out)
            print(f"    Shapefile : {shp_out}")
            print(f"    Centroid  : {centroid_lon}, {centroid_lat}")
        except Exception as e:
            print(f"    ⚠ Shapefile export failed: {e}")

        # ── Plot ───────────────────────────────────────────────────────
        try:
            hillshade_full  = make_hillshade(full_dem_view)
            hillshade_catch = make_hillshade(catchment_dem)
            boundary        = catch_mask & ~binary_erosion(catch_mask)
            bnd_rgba        = np.zeros((*boundary.shape, 4))
            bnd_rgba[boundary] = [1, 0.2, 0.1, 0.9]

            v_slope = slope_deg[~np.isnan(slope_deg)]
            s_flat  = (v_slope <  2).sum()
            s_gent  = ((v_slope >=  2) & (v_slope < 5)).sum()
            s_mod   = ((v_slope >=  5) & (v_slope < 15)).sum()
            s_step  = ((v_slope >= 15) & (v_slope < 30)).sum()
            s_vst   = (v_slope >= 30).sum()

            stream_rgba = np.zeros((*stream_net.shape, 4))
            stream_rgba[stream_net] = [0.0, 0.3, 1.0, 0.85]

            fig, axes = plt.subplots(1, 3, figsize=(22, 7))
            fig.suptitle(
                f"{station_name}  |  Area={area_km2:.1f} km²  |  "
                f"Mean Elev={mean_elev:.0f} m  |  Slope={mean_slope:.1f}°  |  "
                f"Dd={drain_density:.3f} km/km²",
                fontsize=11, fontweight="bold"
            )

            ax = axes[0]
            ax.set_title("Full DEM + Catchment Boundary", fontsize=10)
            ax.imshow(hillshade_full, extent=grid.extent,
                      cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            im1 = ax.imshow(full_dem_view, extent=grid.extent,
                            cmap="terrain", alpha=0.6, interpolation="bilinear")
            ax.imshow(bnd_rgba, extent=grid.extent, interpolation="nearest")
            ax.scatter(lon,    lat,    marker="x", s=80,  c="red",
                       linewidths=2, zorder=5, label="Gauge")
            ax.scatter(x_snap, y_snap, marker="o", s=60,  c="yellow",
                       edgecolors="black", linewidths=1, zorder=6, label="Snapped")
            plt.colorbar(im1, ax=ax, fraction=0.035,
                         pad=0.04).set_label("Elevation (m)", fontsize=8)
            ax.legend(fontsize=7, loc="upper left")
            ax.set_xlabel("Longitude (°E)", fontsize=8)
            ax.set_ylabel("Latitude (°N)",  fontsize=8)

            ax = axes[1]
            ax.set_title("Catchment DEM + Stream Network", fontsize=10)
            ax.imshow(hillshade_catch, extent=grid.extent,
                      cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            im2 = ax.imshow(catchment_dem, extent=grid.extent,
                            cmap="terrain", alpha=0.55, interpolation="bilinear")
            ax.imshow(stream_rgba, extent=grid.extent, interpolation="nearest")
            ax.imshow(bnd_rgba,    extent=grid.extent, interpolation="nearest")
            plt.colorbar(im2, ax=ax, fraction=0.035,
                         pad=0.04).set_label("Elevation (m)", fontsize=8)
            stats = (
                f"Area      : {area_km2:.1f} km²\n"
                f"Min elev  : {min_elev:.0f} m\n"
                f"Max elev  : {max_elev:.0f} m\n"
                f"Mean elev : {mean_elev:.0f} m\n"
                f"Relief    : {relief:.0f} m\n"
                f"Slope     : {mean_slope:.1f}°\n"
                f"Stream len: {stream_len_km:.1f} km\n"
                f"Dd        : {drain_density:.3f} km/km²"
            )
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
            ax.set_xlabel("Longitude (°E)", fontsize=8)
            ax.set_ylabel("Latitude (°N)",  fontsize=8)

            ax = axes[2]
            ax.set_title("Slope Class Distribution", fontsize=10)
            labels = ["Flat\n(<2°)", "Gentle\n(2–5°)", "Moderate\n(5–15°)",
                      "Steep\n(15–30°)", "Very Steep\n(>30°)"]
            sizes  = [s_flat, s_gent, s_mod, s_step, s_vst]
            colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
            sizes_f = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
            if sizes_f:
                sz, lb, cl = zip(*sizes_f)
                ax.pie(sz, labels=lb, colors=cl, autopct="%1.1f%%",
                       startangle=140, textprops={"fontsize": 8},
                       wedgeprops={"edgecolor": "white", "linewidth": 1.2})

            plt.tight_layout()
            plot_out = os.path.join(DIR_PLOTS, f"{safe}.png")
            plt.savefig(plot_out, dpi=110, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"    ⚠ Plot failed: {e}")
            plt.close("all")

        # ── Fill result ────────────────────────────────────────────────
        result.update(
            status         = "SUCCESS",
            area_km2       = round(area_km2,       2),
            mean_elev_m    = round(mean_elev,       2),
            min_elev_m     = round(min_elev,        2),
            max_elev_m     = round(max_elev,        2),
            relief_m       = round(relief,          2),
            mean_slope_deg = round(mean_slope,      2),
            drain_density  = round(drain_density,   4),
            stream_len_km  = round(stream_len_km,   2),
            centroid_lon   = centroid_lon,
            centroid_lat   = centroid_lat,
            snap_lon       = round(x_snap,          5),
            snap_lat       = round(y_snap,          5),
            snap_dist_km   = round(snap_dist_km,    3),
            final_buffer   = buffer,
            error          = None,
        )
        return result

    result["error"] = f"Catchment still cut at max_buffer={MAX_BUFFER}°"
    return result


# =============================================================================
# BATCH LOOP
# =============================================================================

excel_files = sorted(glob.glob(os.path.join(EXCEL_FOLDER, "*.xlsx")))
total       = len(excel_files)

csv_path = os.path.join(DIR_CSV, "All_Stations_morphometry.csv")
if os.path.exists(csv_path):
    done_df      = pd.read_csv(csv_path)
    completed    = set(done_df[done_df["status"] == "SUCCESS"]["file"].tolist())
    all_results  = done_df.to_dict("records")
    print(f"Resuming — {len(completed)} stations already completed")
else:
    completed   = set()
    all_results = []

failed = []

print(f"\nFound {total} Excel files")
print(f"Output root : {OUTPUT_ROOT}")
print(f"Mosaic DEM  : {MOSAIC_PATH}")
print("=" * 65)

for i, excel_path in enumerate(excel_files):
    filename = os.path.basename(excel_path)

    if filename in completed:
        print(f"[{i+1}/{total}] ✓ Already done — {filename}")
        continue

    print(f"\n[{i+1}/{total}]  {filename}")
    station_name, lat, lon = get_station_info(excel_path)

    if station_name is None or lat is None or lon is None:
        print("  ⚠ Could not read metadata — skipping")
        failed.append({"file": filename, "error": "metadata read failed",
                       "station": "unknown"})
        continue

    print(f"  Station : {station_name}")
    print(f"  Lat/Lon : {lat}, {lon}")

    annual_max = get_annual_max(excel_path)
    n_years    = len(annual_max) if annual_max is not None else 0
    print(f"  Years of data: {n_years}")

    try:
        res             = delineate_station(station_name, lat, lon)
        res["file"]     = filename
        res["n_years"]  = n_years
        all_results.append(res)

        if res["status"] == "FAILED":
            failed.append({"file": filename, "station": station_name,
                           "error": res["error"]})
            print(f"  ✗ FAILED: {res['error']}")
        else:
            print(f"  ✓ SUCCESS | area={res['area_km2']} km² | "
                  f"centroid=({res['centroid_lon']}, {res['centroid_lat']})")

    except Exception as e:
        print(f"  ✗ Unexpected crash: {e}")
        traceback.print_exc()
        failed.append({"file": filename, "station": station_name, "error": str(e)})
        all_results.append(dict(
            station=station_name, lat=lat, lon=lon,
            file=filename, n_years=n_years,
            status="CRASHED", error=str(e),
            area_km2=None, mean_elev_m=None, min_elev_m=None,
            max_elev_m=None, relief_m=None, mean_slope_deg=None,
            drain_density=None, stream_len_km=None,
            centroid_lon=None, centroid_lat=None,
            snap_lon=None, snap_lat=None, snap_dist_km=None,
            final_buffer=None,
        ))

    csv_path = os.path.join(DIR_CSV, "All_Stations_morphometry.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
success = sum(1 for r in all_results if r.get("status") == "SUCCESS")

print("\n" + "=" * 65)
print("BATCH COMPLETE")
print("=" * 65)
print(f"Total    : {total}")
print(f"Success  : {success}")
print(f"Failed   : {len(failed)}")
print(f"\nResults CSV : {csv_path}")
print(f"Shapefiles  : {DIR_SHP}")
print(f"Plots       : {DIR_PLOTS}")

failed_csv = os.path.join(DIR_CSV, "failed_stations.csv")
if failed:
    print(f"\nFailed stations ({len(failed)}):")
    for f in failed:
        print(f"  {f.get('station','?')} — {f['error']}")
    pd.DataFrame(failed).to_csv(failed_csv, index=False)
    print(f"\nFailed list : {failed_csv}")