# Data Characteristics
- single sample consists of a 512 m x 512 m patch of each data source
- 128 x 128 input size at input layer of backbone networks
    - 4 m pixel size for Sentinel-2, Planet, SRTM data
- 32 x 32 input size into regression head
    - 16 m pixel size for wind data
- target CRS: `EPSG:31468` for all raster data

# Sentinel-2
**Original format:**
- scenes containing 13 bands in individual files
- pixel size 10/20/30 m depending on band
- in local projection, may vary between scenes
- uint16

**Processing:**
- resample each band to 4 m pixel size using nearest neighbor sampling
- compose individual bands into single raster file containing 13 channels
- reproject into target CRS
- merge all files sharing the same timestamp

# Planet
**Original format:**
- scenes containing 4 bands in a single file
- pixel size 3 m
- in `EPSG:4236` (WGS84 lat/lon)
- uint16

**Processing:**
- resample to 4 m pixel size using nearest neighbor sampling
- reproject into target CRS
- merge all files sharing the same timestamp

# DGM 1 Data (DEM)
**Original format:**
- tiles arranged in 1 km x 1 km grid
- in `EPSG:25832` (UTM 32 N)
- float32

**Processing:**
- select and merge desired tiles
- reproject and resample to target CRS and resolution

# Wind Data
**Original format:**
- .csv files containing station locations, measurements, timestamps etc.
- various formats depending on country

**Processing:**
- retrieve wind speed and location from all available stations for a given date
- transform station positions to target CRS
- create raster of target resolution over region of interest
- inside convex hull around station positions: interpolate wind speed linearly at grid points
- outside convex hull around station positions: nearest-neighbor interpolation of wind speed at grid points
- float64

# Damage Data
**Original format:**
- shapefile containing damaged areas as polygons
- in `EPSG:31468`

**Processing:**
- rasterize into binary pixel mask at target resolution
- uint8

# ~SRTM Data (DEM)~
**~Original format:~**
- rectangular tile containing elevation values
- 30 m pixel size
- in `EPSG:4236` (WGS84 lat/lon)

**~Processing:~**
- resample to 4 m pixel size using nearest neighbor sampling
- reproject into target CRS)~
