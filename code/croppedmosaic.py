import rasterio
from rasterio.windows import from_bounds

with rasterio.open(r"D:\Manasa_ce24b050\Chingka_lai\India_DEM\india_dem_mosaic.tif") as src:
    window = from_bounds(68, 8, 88, 28, src.transform)
    data = src.read(window=window)
    transform = src.window_transform(window)
    meta = src.meta.copy()
    meta.update({"height": data.shape[1], "width": data.shape[2], "transform": transform})
    with rasterio.open(r"D:\Manasa_ce24b050\Chingka_lai\India_DEM\india_dem_cropped.tif", "w", **meta) as dst:
        dst.write(data)
print("Cropped mosaic saved!")