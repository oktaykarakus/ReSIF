import rasterio
from rasterio.merge import merge
import glob
import multiprocessing as mp

dirs = glob.glob('D:\\Nino\\Wind Throw Data\\sentinel\\*_Passau\\resampled\\') + glob.glob('D:\\Nino\\Wind Throw Data\\planet\\*_Passau\\resampled\\')

def merge_files(directory):
    files = glob.glob(directory + '*.tif')
    with rasterio.open(files[0], 'r') as file:
        crs_out = file.crs
    
    merged, trans = merge(files, nodata=0, target_aligned_pixels=True)
    
    out_name = files[0].replace('\\resampled\\', '\\').replace('.tif', '_merged.tif')
    with rasterio.open(out_name,
                       mode='w',
                       driver='GTiff',
                       width=merged.shape[2],
                       height=merged.shape[1],
                       count=merged.shape[0],
                       dtype=merged.dtype,
                       transform=trans,
                       crs=crs_out) as outfile:
        outfile.write(merged)

if __name__=='__main__':  
    with mp.Pool() as pool:
        pool.map(merge_files, dirs)