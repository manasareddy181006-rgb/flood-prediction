import geopandas as gpd
import glob
import os

shp_files = glob.glob(r"d:\Manasa_ce24b050\Chingka_lai\Output\Shapefiles\*.shp")
gdfs = [gpd.read_file(f) for f in shp_files]
all_catch = gpd.pd.concat(gdfs, ignore_index=True)
all_catch = all_catch.set_crs("EPSG:4326", allow_override=True)

problems = []
for i, row1 in all_catch.iterrows():
    for j, row2 in all_catch.iterrows():
        if i >= j: continue
        if row1.geometry.intersects(row2.geometry):
            overlap = row1.geometry.intersection(row2.geometry).area
            smaller = min(row1.geometry.area, row2.geometry.area)
            pct = overlap / smaller * 100
            if pct > 10 and pct < 99:  # partial overlap = bad
                problems.append((row1.station, row2.station, round(pct, 1)))
                print(f"BAD OVERLAP: {row1.station} ↔ {row2.station} ({pct:.1f}%)")

print(f"\nTotal bad overlaps: {len(problems)}")
import subprocess

bad_stations = set()
for s1, s2, pct in problems:
    bad_stations.add(s1)
    bad_stations.add(s2)

print(f"\nStations to rerun with tight snap: {bad_stations}")
# {'Hommaragalli', 'Khadka', 'Muthankera', 'T. narasipur', 'Koli(bk)'}