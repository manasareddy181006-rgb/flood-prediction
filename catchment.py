!pip install pysheds pandas openpyxl matplotlib


import pandas as pd
import numpy as np
# 1. LOAD DATA
# We skip 7 rows to get past the headers
df = pd.read_excel('River Water Discharge_Muthankera.xlsx', sheet_name=2)

# --- DIAGNOSTIC PRINT ---
# This helps us see if we are looking at the right columns
print("Check: First 5 rows of what Python sees:")
print(df.iloc[:5, [2, 4]]) # Shows 3rd and 5th columns
# ------------------------

# 2. CONVERT DATA
# We use iloc[:, 2] for Date and iloc[:, 4] for Discharge (Value)
df['Date'] = pd.to_datetime(df.iloc[:, 2], errors='coerce')
df['Flow'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')

# 3. CLEANING
# Drop rows that don't have a date or a flow value
df_clean = df.dropna(subset=['Date', 'Flow']).copy()

# 4. GROUP BY YEAR
if not df_clean.empty:
    df_clean['Year'] = df_clean['Date'].dt.year
    # We filter for years > 1900 to avoid weird Excel date errors
    df_clean = df_clean[df_clean['Year'] > 1900]
    annual_max = df_clean.groupby('Year')['Flow'].max()
    
    print("\n--- SUCCESS! ANNUAL MAX SERIES ---")
    print(annual_max.head(20))
else:
    print("\n--- ERROR: No data found! ---")
    print("Columns available are:", df.columns.tolist())
    print("Try changing iloc[:, 4] to iloc[:, 3] in the code if 'Flow' is empty.")

# --- PART 2: THE MAP (THE FIX FOR YELLOW SQUARE) ---
grid = Grid.from_raster('output_SRTMGL1.tif')
dem = grid.read_raster('output_SRTMGL1.tif')

# Conditioning
pit_filled = grid.fill_pits(dem)
flooded = grid.fill_depressions(pit_filled)
resolved = grid.resolve_flats(flooded)

# Flow Direction and Accumulation
fdir = grid.flowdir(resolved)
acc = grid.accumulation(fdir)

# THE SNAP: We use a larger snap (500) to make sure we hit the river
x_snap, y_snap = grid.snap_to_mask(acc > 500, (76.0842, 11.8081))

# Delineate
catchment = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='label')
# Plotting
grid.clip_to(catchment)
plt.figure(figsize=(10, 8))
plt.imshow(grid.view(dem), extent=grid.extent, cmap='terrain')
plt.title("Muthankera Catchment Boundary")
plt.colorbar(label='Elevation (m)')
plt.show()

# Final Attribute
area_km2 = (catchment.sum() * 900) / 1e6
print(f"Total Catchment Area: {area_km2:.2f} km²")