import os
import requests
import pandas as pd
import numpy as np
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

# Patch for newer NumPy versions
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "1024da750b3a91f41b929e8024b0a1e0"
lon = 76.08419
lat = 11.80811
folder = r"C:\Users\bsnre\OneDrive\Desktop\Chingka_lai"

excel_path = os.path.join(folder, "River Water Discharge_Muthankera.xlsx")

# -----------------------------
# BUFFER / DELINEATION PARAMS
# -----------------------------
buffer      = 0.5
max_buffer  = 5.0
buffer_step = 0.25
MIN_AREA_KM2 = 50

# -----------------------------
# LOAD DISCHARGE DATA
# -----------------------------
df = pd.read_excel(excel_path, sheet_name=1, skiprows=6)
df.columns = df.columns.astype(str).str.strip()

print("Columns:", df.columns.tolist())

df["Date"] = pd.to_datetime(
    df["Data Time"].astype(str).str.strip(),
    format="%Y-%m-%dT%H:%M:%S",
    errors="coerce"
)
df["Flow"] = pd.to_numeric(df["Data Value"], errors="coerce")

print(df[["Data Time", "Date", "Data Value", "Flow"]].head())

df_clean = df.dropna(subset=["Date", "Flow"]).copy()
df_clean["Year"] = df_clean["Date"].dt.year

annual_max = (
    df_clean[df_clean["Year"] >= 1973]
    .groupby("Year")["Flow"]
    .max()
)

print("\nAnnual Max Series (first 5):")
print(annual_max.head())

# -----------------------------
# FUNCTION: DOWNLOAD DEM
# -----------------------------
def download_dem(buffer, dem_path):
    url = (
        f"https://portal.opentopography.org/API/globaldem"
        f"?demtype=SRTMGL1"
        f"&south={lat - buffer}"
        f"&north={lat + buffer}"
        f"&west={lon - buffer}"
        f"&east={lon + buffer}"
        f"&outputFormat=GTiff"
        f"&API_Key={API_KEY}"
    )
    print(f"\nDownloading DEM | buffer = {buffer:.2f}°")
    response = requests.get(url)
    if response.status_code == 200:
        with open(dem_path, "wb") as f:
            f.write(response.content)
        print(f"Saved: {dem_path}")
        return True
    else:
        print(f"Download failed — HTTP {response.status_code}")
        print(response.text)
        return False

# -----------------------------
# FUNCTION: BOUNDARY CHECK
# -----------------------------
def catchment_touches_boundary(catchment_array):
    top    = np.any(catchment_array[0, :])
    bottom = np.any(catchment_array[-1, :])
    left   = np.any(catchment_array[:, 0])
    right  = np.any(catchment_array[:, -1])
    edge_count = (catchment_array[0, :].sum() + catchment_array[-1, :].sum() +
                  catchment_array[:, 0].sum() + catchment_array[:, -1].sum())
    print(f"  Boundary — Top:{top} Bottom:{bottom} Left:{left} Right:{right} | Edge pixels:{edge_count}")
    return top or bottom or left or right

# -----------------------------
# FUNCTION: HILLSHADE
# -----------------------------
def make_hillshade(dem_array, azimuth=315, altitude=45):
    az  = np.radians(azimuth)
    alt = np.radians(altitude)
    dy_arr, dx_arr = np.gradient(np.where(np.isnan(dem_array), 0, dem_array))
    slope  = np.arctan(np.sqrt(dx_arr**2 + dy_arr**2))
    aspect = np.arctan2(-dy_arr, dx_arr)
    shaded = (np.sin(alt) * np.cos(slope) +
              np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shaded = np.clip(shaded, 0, 1)
    shaded[np.isnan(dem_array)] = np.nan
    return shaded

# -----------------------------
# AUTOMATIC BUFFER LOOP
# -----------------------------
final_grid = final_dem = final_catchment = None
final_x_snap = final_y_snap = final_buffer = None

while buffer <= max_buffer:

    dem_path = os.path.join(folder, f"dem_buffer_{buffer:.2f}.tif")

    if not os.path.exists(dem_path):
        success = download_dem(buffer, dem_path)
        if not success:
            print("Stopping: DEM download failed.")
            break
    else:
        print(f"\nUsing cached DEM: {dem_path}")

    # Condition DEM
    grid = Grid.from_raster(dem_path)
    dem  = grid.read_raster(dem_path)

    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    resolved   = grid.resolve_flats(flooded)

    fdir = grid.flowdir(resolved)
    acc  = grid.accumulation(fdir)

    print(f"Buffer={buffer:.2f}° | DEM shape:{np.array(dem).shape} | Max acc:{np.nanmax(acc):.0f}")

    # -----------------------------
    # SMART OUTLET SNAPPING
    # -----------------------------
    acc_array = np.array(acc)
    dx = abs(grid.affine.a)
    dy = abs(grid.affine.e)

    col_out = int((lon - grid.extent[0]) / dx)
    row_out = int((grid.extent[3] - lat) / dy)

    # 1 km search window
    px = int(0.009 / dx)
    r0 = max(0, row_out - px)
    r1 = min(acc_array.shape[0], row_out + px)
    c0 = max(0, col_out - px)
    c1 = min(acc_array.shape[1], col_out + px)

    window = acc_array[r0:r1, c0:c1]
    print(f"  Max acc in 1km window : {window.max():.0f}")
    print(f"  Mean acc in 1km window: {window.mean():.0f}")

    # Try thresholds from high to low — pick closest cell above threshold
    found_snap = False
    x_snap_final = y_snap_final = None
    catchment = catch_view = None
    area_km2 = 0

    for threshold in [500000, 100000, 50000, 10000, 1000]:
        mask = window >= threshold
        if mask.sum() == 0:
            continue

        rows_t, cols_t = np.where(mask)
        dists = (rows_t - (row_out - r0))**2 + (cols_t - (col_out - c0))**2
        best  = np.argmin(dists)

        best_row = r0 + rows_t[best]
        best_col = c0 + cols_t[best]

        x_snap = grid.extent[0] + (best_col + 0.5) * dx
        y_snap = grid.extent[3] - (best_row + 0.5) * dy
        snap_dist_km = ((x_snap - lon)**2 + (y_snap - lat)**2)**0.5 * 111

        print(f"  Threshold {threshold:>7}: acc={acc_array[best_row, best_col]:.0f} "
              f"| dist={snap_dist_km:.3f} km")

        catchment_test = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype="coordinate")
        catch_test     = grid.view(catchment_test).astype(bool)
        area_test      = (catch_test.sum() * 900) / 1e6
        print(f"    → Catchment area: {area_test:.2f} km²")

        if area_test >= MIN_AREA_KM2:
            print(f"  ✓ Using threshold {threshold}")
            x_snap_final  = x_snap
            y_snap_final  = y_snap
            catchment     = catchment_test
            catch_view    = catch_test
            area_km2      = area_test
            found_snap    = True
            break

    if not found_snap:
        print(f"  ⚠ No valid threshold found — increasing buffer...")
        buffer = round(buffer + buffer_step, 2)
        continue

    print(f"\n  Original outlet  : {lon}, {lat}")
    print(f"  Snapped outlet   : {x_snap_final:.5f}, {y_snap_final:.5f}")
    print(f"  Catchment area   : {area_km2:.2f} km²")

    # Boundary check
    is_cut = catchment_touches_boundary(catch_view)

    if not is_cut:
        print(f"\n✓ Catchment fully contained. Final buffer = {buffer:.2f}°")
        final_grid      = grid
        final_dem       = dem
        final_catchment = catchment
        final_x_snap    = x_snap_final
        final_y_snap    = y_snap_final
        final_buffer    = buffer
        break
    else:
        print(f"✗ Catchment cut at boundary. Increasing buffer to {buffer + buffer_step:.2f}°...")

    buffer = round(buffer + buffer_step, 2)

if final_grid is None:
    print(f"\n⚠ Could not delineate within max_buffer={max_buffer}°.")
    print("Try: 1) Increase max_buffer  2) Reduce MIN_AREA_KM2  3) Check outlet coordinates")
    exit()

# -----------------------------
# PLOT FINAL RESULT
# -----------------------------
grid      = final_grid
dem       = final_dem
catchment = final_catchment

full_dem_view = grid.view(dem).astype(float)
full_dem_view[full_dem_view <= 0] = np.nan

catch_mask    = grid.view(catchment).astype(bool)
catch_view    = catch_mask.astype(float)
catch_view[catch_view == 0] = np.nan

catchment_dem = full_dem_view.copy()
catchment_dem[~catch_mask] = np.nan

hillshade_full  = make_hillshade(full_dem_view)
hillshade_catch = make_hillshade(catchment_dem)

boundary      = catch_mask & ~binary_erosion(catch_mask)
boundary_rgba = np.zeros((*boundary.shape, 4))
boundary_rgba[boundary] = [1, 0.2, 0.1, 0.9]

area_km2   = (catch_mask.sum() * 900) / 1e6
valid_elev = catchment_dem[~np.isnan(catchment_dem)]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    f"Muthankera Catchment — Buffer = {final_buffer:.2f}° | Area = {area_km2:.1f} km²",
    fontsize=14, fontweight="bold"
)

# ── LEFT: Full DEM + catchment overlay ────────────────────────────────────
ax = axes[0]
ax.set_title("Full DEM with Catchment Boundary", fontsize=12)
ax.imshow(hillshade_full, extent=grid.extent, cmap="gray",
          vmin=0, vmax=1, interpolation="bilinear")
im_dem = ax.imshow(full_dem_view, extent=grid.extent, cmap="terrain",
                   alpha=0.6, interpolation="bilinear")
ax.imshow(catch_view, extent=grid.extent, cmap="Blues",
          alpha=0.3, interpolation="nearest")
ax.imshow(boundary_rgba, extent=grid.extent, interpolation="nearest")
ax.scatter(lon, lat, marker="x", s=100, c="red", linewidths=2,
           zorder=5, label=f"Original ({lon}, {lat})")
ax.scatter(final_x_snap, final_y_snap, marker="o", s=80, c="yellow",
           edgecolors="black", linewidths=1, zorder=6,
           label=f"Snapped ({final_x_snap:.4f}, {final_y_snap:.4f})")
plt.colorbar(im_dem, ax=ax, fraction=0.035, pad=0.04).set_label("Elevation (m)", fontsize=10)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.legend(fontsize=8, loc="upper left")

# ── RIGHT: Catchment DEM clipped ──────────────────────────────────────────
ax = axes[1]
ax.set_title("Catchment DEM (Clipped)", fontsize=12)
ax.imshow(hillshade_catch, extent=grid.extent, cmap="gray",
          vmin=0, vmax=1, interpolation="bilinear")
im_catch = ax.imshow(catchment_dem, extent=grid.extent, cmap="terrain",
                     alpha=0.65, interpolation="bilinear")
ax.imshow(boundary_rgba, extent=grid.extent, interpolation="nearest")
ax.scatter(lon, lat, marker="x", s=100, c="red", linewidths=2, zorder=5)
ax.scatter(final_x_snap, final_y_snap, marker="o", s=80, c="yellow",
           edgecolors="black", linewidths=1, zorder=6)

stats_text = (
    f"Min elev : {valid_elev.min():.0f} m\n"
    f"Max elev : {valid_elev.max():.0f} m\n"
    f"Mean elev: {valid_elev.mean():.0f} m\n"
    f"Relief   : {valid_elev.max() - valid_elev.min():.0f} m"
)
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
plt.colorbar(im_catch, ax=ax, fraction=0.035, pad=0.04).set_label("Elevation (m)", fontsize=10)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

plt.tight_layout()
out_path = os.path.join(folder, f"catchment_dem_buffer_{final_buffer:.2f}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out_path}")
plt.show()

# -----------------------------
# FINAL SUMMARY
# -----------------------------
print("\n========== RESULTS ==========")
print(f"Final buffer     : {final_buffer:.2f}°")
print(f"Snapped outlet   : {final_x_snap:.5f}, {final_y_snap:.5f}")
print(f"Catchment area   : {area_km2:.2f} km²")
print(f"Min elevation    : {valid_elev.min():.0f} m")
print(f"Max elevation    : {valid_elev.max():.0f} m")
print(f"Mean elevation   : {valid_elev.mean():.0f} m")
print(f"Relief           : {valid_elev.max() - valid_elev.min():.0f} m")
print("=============================")

# Add at the very end of your script
print("\n========== CROSS CHECK ==========")
print(f"Catchment area    : {area_km2:.1f} km²")
print(f"Snapped outlet    : {final_x_snap:.5f}, {final_y_snap:.5f}")
print(f"Original outlet   : {lon}, {lat}")
print(f"Snap distance     : {((final_x_snap-lon)**2 + (final_y_snap-lat)**2)**0.5 * 111:.3f} km")
print("")
print("Expected ranges for common Kabini sub-catchments:")
print("  Muthankera gauge  : ~700–900 km²  (if on tributary)")
print("  Kabini at Beechanahalli : ~2200 km²")
print("  Kabini at Mananthavady  : ~1150 km²")
print("")
if area_km2 > 900:
    print("⚠ Area > 900 km² — outlet may be on main Kabini stem, not Muthankera tributary")
    print("  Try reducing threshold to force snap to smaller tributary:")
    print("  Change MIN_AREA_KM2 = 50 and add max area check: area_test <= 900")
elif 200 <= area_km2 <= 900:
    print("✓ Area looks reasonable for a Kabini sub-catchment gauge")
else:
    print("⚠ Area seems too small — check outlet location")