import numpy as np
from itertools import product
import rasterio
import pyproj

def get_patchdict(bounds, patchsize, overlap):
    xmin, ymin, xmax, ymax = bounds
    
    if isinstance(patchsize, tuple):
        sizex, sizey = patchsize
    else:
        sizex = sizey = patchsize
        
    if isinstance(overlap, tuple):
        overlapx, overlapy = overlap
    else:
        overlapx = overlapy = overlap
        
    leftx = np.arange(xmin, xmax, (1-overlapx) * sizex)
    rightx = np.arange(0, len(leftx)) * ((1-overlapx) * sizex) + (xmin + sizex)
    
    lowery = np.arange(ymin, ymax, (1-overlapy) * sizey)
    uppery = np.arange(0, len(lowery)) * ((1-overlapy) * sizey) + (ymin + sizey)
    
    lowerleft = list(product(leftx, lowery))
    upperright = list(product(rightx, uppery))
    
    patchdict = dict()
    keylen = len(str(len(lowerleft)))
    for idx, (ll, ur) in enumerate(zip(lowerleft, upperright)):
        key = '{0:0>{keylen}}'.format(idx, keylen=keylen)
        patchbounds = ll + ur
        patchdict[key] = patchbounds
    
    return patchdict

def chop_raster(file, patchdict, dest_folder, suffix, force_size=None):
    if force_size is not None:
        if isinstance(force_size, tuple):
            w, h = force_size
        else:
            w = h = force_size
                    
    with rasterio.open(file, 'r') as raster:
        for key, (xmin, ymin, xmax, ymax) in patchdict.items():
            
            crs_proj = pyproj.crs.CRS.from_user_input(raster.crs)
            if (crs_proj.axis_info[0].direction == 'east') and (crs_proj.axis_info[1].direction == 'north'):
                # Easting, Northing axis order
                west_south_east_north = (xmin, ymin, xmax, ymax)
            elif (crs_proj.axis_info[0].direction == 'north') and (crs_proj.axis_info[1].direction == 'east'):
                # Northing, Easting axis order
                west_south_east_north = (ymin, xmin, ymax, xmax)
            else:
                raise RuntimeError('unexpected axis order or direction in crs_proj:\n' + crs_proj.__repr__())
            
            window = rasterio.windows.from_bounds(*west_south_east_north, transform = raster.transform)
            patchtrans = raster.window_transform(window)
            patchdata = raster.read(window=window)
            
            if force_size is not None:
                if patchdata.shape[1:] != (w, h):
                    full_patch = np.zeros((raster.count, w, h), dtype=patchdata.dtype)
                    colstart = colstop = rowstart = rowstop = None
                    
                    if patchdata.shape[2] != w:
                        if (window.col_off < 0) and (0 < window.col_off + window.width < raster.width):
                            # align data right in full_patch array
                            colstart, colstop = (- patchdata.shape[2], None)
                        elif (window.col_off + window.width > raster.width) and (0 < window.col_off < raster.width):
                            # align data left in full_patch array
                            colstart, colstop = (None, patchdata.shape[2])
                        else:
                            raise RuntimeError(f'confusing window fuckery with window {window.flatten()} and raster shape {raster.shape}')

                    if patchdata.shape[1] != h:
                        if (window.row_off < 0) and (0 < window.row_off + window.height < raster.height):
                            # align data bottom in full_patch array
                            rowstart, rowstop = (- patchdata.shape[1], None)
                            
                        elif (window.row_off + window.height > raster.height) and (0 < window.row_off < raster.height):
                            # align data top in full_patch array
                            rowstart, rowstop = (None, patchdata.shape[1])
                        else:
                            raise RuntimeError(f'confusing window fuckery with window {window.flatten()} and raster shape {raster.shape}')

                    full_patch[:, rowstart:rowstop, colstart:colstop] = patchdata
                    patchtrans = rasterio.transform.from_bounds(*west_south_east_north, full_patch.shape[2], full_patch.shape[1])
                    patchdata = full_patch
            
            patchname = dest_folder + key + suffix + '.tif'
            with rasterio.open(patchname,
                                mode='w',
                                driver='GTiff',
                                width=patchdata.shape[2],
                                height=patchdata.shape[1],
                                count=patchdata.shape[0],
                                dtype=patchdata.dtype,
                                transform=patchtrans,
                                crs=raster.crs) as patchfile:
                  patchfile.write(patchdata)
            
patchdict = get_patchdict((5388000, 4607500, 5398000, 4624000), 512, 0.5)

with open('D:/Nino/Wind Throw Data/patches.txt', 'w') as file:
    header = ['idx,xmin,ymin,xmax,ymax\n']
    patches = [f'{idx},{xmin},{ymin},{xmax},{ymax}\n' for idx, (xmin, ymin, xmax, ymax) in patchdict.items()]
    file.writelines(header+patches)

chop_raster('D:/Nino/Wind Throw Data/patches/out_5387500_4607000_2017-08-18.tif', patchdict, 'D:/Nino/Wind Throw Data/patches/wind/', '_wind', force_size=32)
