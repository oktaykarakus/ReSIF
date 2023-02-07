import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject
from itertools import product
import pyproj

bounds_target = (5387500, 4607000, 5398500, 4624500) # in target CRS
crs_in = pyproj.CRS("epsg:25832")
crs_target = pyproj.CRS("epsg:31468")
folder = 'D:\\Nino\\Wind Throw Data\\dgm1\\'
res_target = (4, 4)
out_name = 'D:\\Nino\\Wind Throw Data\\dgm_merged.tif'

transformer = pyproj.transformer.Transformer.from_crs(crs_target, crs_in)
bounds_in = transformer.transform(bounds_target[0], bounds_target[1]) + transformer.transform(bounds_target[2], bounds_target[3])

bounds_in = tuple(int(coord // 1000) for coord in bounds_in)
XYs = list(product(range(bounds_in[0], bounds_in[2] + 1), range(bounds_in[1], bounds_in[3] + 1)))

files = [folder + f'{X}_{Y}.tif' for X, Y in XYs]

data, trans = merge(files, nodata=99999, target_aligned_pixels=True)
data_out, transform_out = reproject(data, dst_resolution=res_target, src_transform=trans, src_crs=rasterio.crs.CRS.from_user_input(crs_in), dst_crs=rasterio.crs.CRS.from_user_input(crs_target))

with rasterio.open(out_name,
                   mode='w',
                   driver='GTiff',
                   width=data_out.shape[2],
                   height=data_out.shape[1],
                   count=data_out.shape[0],
                   dtype=data_out.dtype,
                   transform=transform_out,
                   crs=crs_target) as outfile:
    outfile.write(data_out)
