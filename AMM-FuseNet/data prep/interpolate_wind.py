import pyproj
import pandas
import numpy as np
import rasterio
from itertools import product
from scipy.interpolate import griddata
import glob
import windgetter as wg

date = '2017-08-18'

x_min, y_min, x_max, y_max = (5387500,4607000,5398500,4624500)

crs_out = pyproj.CRS("epsg:31468")
resolution = 16    

dwd_sdo_file = 'D:/Nino/Wind Throw Data/wind/dwd/data/sdo_OBS_DEU_P1D_F_X.csv'
dwd_data_file = 'D:/Nino/Wind Throw Data/wind/dwd/data/data_OBS_DEU_P1D_F_X.csv'

cz_files = glob.glob('D:/Nino/Wind Throw Data/wind/czechia/*.csv')

at_data_file = 'D:/Nino/Wind Throw Data/wind/austria/TAG Datensatz_19800101_20230130.csv'
at_station_file = 'D:/Nino/Wind Throw Data/wind/austria/TAG Stations-Metadaten.csv'


def get_grid_points(x_min, y_min, x_max, y_max, resolution):
    XX = np.arange(x_min, x_max, resolution)
    YY = np.arange(y_min, y_max, resolution)
    nodes = np.array(list(product(XX, YY)))
    grid_bounds = (XX[0], YY[0], XX[-1], YY[-1])
    return nodes, grid_bounds

def interpolate(station_coords, values, gridpoints):
    vals = griddata(station_coords, values, gridpoints, method='linear', fill_value=np.nan)
    vals[np.isnan(vals)] = griddata(station_coords, values, gridpoints[np.isnan(vals)], method='nearest')
    return vals

def to_image(vals, crs_out, grid_bounds, resolution):
    n_col = int(np.ceil((x_max - x_min)/resolution))
    n_row = int(np.ceil((y_max - y_min)/resolution))
    assert n_col * n_row == len(vals)
    
    if (crs_out.axis_info[0].direction == 'east') and (crs_out.axis_info[1].direction == 'north'):
        # Easting, Northing axis order
        img = vals.reshape(n_col, n_row).T[::-1,:]
        west, north = grid_bounds[0], grid_bounds[3]
    elif (crs_out.axis_info[0].direction == 'north') and (crs_out.axis_info[1].direction == 'east'):
        # Northing, Easting axis order
        img = vals.reshape(n_col, n_row)[::-1,:]
        west, north = grid_bounds[1], grid_bounds[2]
    else:
        raise RuntimeError('unexpected axis order or direction in crs_out:\n' + crs_out.__repr__())
    
    transform = rasterio.transform.from_origin(west, north, resolution, resolution)
    # transform = rasterio.transform.from_bounds(*grid_bounds, img.shape[1], img.shape[0])
    return img, transform


wg_de = wg.WindGetter_DE(dwd_sdo_file, dwd_data_file, crs_out)
coords_de, speeds_de = wg_de.get_measurements(date)

wg_cz = wg.WindGetter_CZ(cz_files, crs_out)
coords_cz, speeds_cz = wg_cz.get_measurements(date)

wg_at = wg.WindGetter_AT(at_station_file, at_data_file, crs_out)
coords_at, speeds_at = wg_at.get_measurements(date)

coords = np.concatenate([coords_de, coords_cz, coords_at])
speeds = np.concatenate([speeds_de, speeds_cz, speeds_at])

grid, grid_bounds = get_grid_points(x_min, y_min, x_max, y_max, resolution)
vals = interpolate(coords, speeds, grid)
img, transform = to_image(vals, crs_out, grid_bounds, resolution)


out_name = f'D:/Nino/Wind Throw Data/wind/testimgs/out_{x_min}_{y_min}_{date}.tif'
with rasterio.open(out_name,
                    mode='w',
                    driver='GTiff',
                    width=img.shape[1],
                    height=img.shape[0],
                    count=1,
                    dtype=img.dtype,
                    transform=transform,
                    crs=crs_out) as outfile:
    outfile.write(img[None, :, :])