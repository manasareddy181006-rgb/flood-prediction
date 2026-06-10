# Run this ONCE after computing acc — before the buffer loop
# Just to visualize accumulation near your outlet

grid_diag = Grid.from_raster(os.path.join(folder, "dem_buffer_0.50.tif"))
dem_diag  = grid_diag.read_raster(os.path.join(folder, "dem_buffer_0.50.tif"))

pit_filled = grid_diag.fill_pits(dem_diag)
flooded    = grid_diag.fill_depressions(pit_filled)
resolved   = grid_diag.resolve_flats(flooded)
fdir_diag  = grid_diag.flowdir(resolved)
acc_diag   = grid_diag.accumulation(fdir_diag)

acc_array = np.array(acc_diag)
dx = abs(grid_diag.affine.a)
dy = abs(grid_diag.affine.e)

# Check accumulation in a small 0.05° window around your outlet
col = int((lon - grid_diag.extent[0]) / dx)
row = int((grid_diag.extent[3] - lat) / dy)
px  = int(0.05 / dx)

window = acc_array[row-px:row+px, col-px:col+px]

# Plot the accumulation map zoomed in near outlet
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(
    np.log1p(window),   # log scale so streams are visible
    extent=[
        lon - 0.05, lon + 0.05,
        lat - 0.05, lat + 0.05
    ],
    cmap="Blues", origin="upper"
)
ax.scatter(lon, lat, marker="x", s=200, c="red", linewidths=2, zorder=5, label="Your outlet")
plt.colorbar(im, label="log(accumulation)")
ax.set_title("Flow Accumulation near Outlet (log scale)\nRed X = your coordinates")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, "accumulation_diagnostic.png"), dpi=150)
plt.show()

# Also print the acc value exactly at your point
print(f"Acc exactly at your outlet ({lon}, {lat}): {acc_array[row, col]:.0f} cells")
print(f"Max acc in 0.05° window: {window.max():.0f} cells")
print(f"If these differ hugely, your coordinates are off the stream.")