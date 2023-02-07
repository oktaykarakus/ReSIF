"""Resamples files to specified CRS, pixel size."""

import rasterio
from rasterio.warp import reproject
from rasterio.crs import CRS
import glob
import multiprocessing as mp

filenames = glob.glob('D:/Nino/Wind Throw Data/planet/*_Passau/files/*AnalyticMS_clip.tif')
crs_out = CRS.from_epsg(31468)
res_out = (4, 4)

def resample(name_in):
        with rasterio.open(name_in) as infile:
            data = infile.read()
            
            # defaults to nearest neighbor resampling
            data_out, transform_out = reproject(data, dst_resolution=res_out, src_transform=infile.transform, src_crs=infile.crs, dst_crs=crs_out)
        
        name_out = name_in.replace('files', 'resampled').replace('.tif', '_resampled.tif')
        with rasterio.open(name_out,
                           mode='w',
                           driver='GTiff',
                           height=data_out.shape[1],
                           width=data_out.shape[2],
                           count=data_out.shape[0],
                           dtype=data_out.dtype,
                           transform=transform_out,
                           crs=crs_out) as outfile:
            outfile.write(data_out)

def main():
    with mp.Pool() as pool:
        pool.map(resample, filenames)
    
if __name__=='__main__':
    main()