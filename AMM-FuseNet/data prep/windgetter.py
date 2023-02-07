import pyproj
import pandas as pd
import csv
import numpy as np
import rasterio

class WindGetter():
    def get_measurements(self, date):
        raise NotImplementedError
        
class WindGetter_DE(WindGetter):
    _crs_in = pyproj.CRS("epsg:4326")
    
    def __init__(self, sdo_file, data_file, crs_out):
        self.sdo_file = sdo_file
        self.data_file = data_file
        self.crs_out = crs_out
        
        self._transformer = pyproj.transformer.Transformer.from_crs(self._crs_in, self.crs_out)
        self.metadata = self.get_metadata()
        return

    def _convert_coords(self, sdo_row):
        lon = float(sdo_row['Geogr_Laenge'].replace(',', '.'))
        lat = float(sdo_row['Geogr_Breite'].replace(',', '.'))
        sdo_row['X'], sdo_row['Y'] = self._transformer.transform(lat, lon)
        return sdo_row
    
    def get_metadata(self):
        metadata = pd.read_csv(self.sdo_file)
        metadata['X'] = np.nan
        metadata['Y'] = np.nan
        metadata = metadata.apply(self._convert_coords, axis=1)
        
        metadata = metadata[['SDO_ID', 'X', 'Y']]
        return metadata
    
    def get_measurements(self, date):
        data_frame = pd.read_csv(self.data_file, usecols=['SDO_ID', 'Zeitstempel', 'Wert'])
        data = data_frame[data_frame['Zeitstempel']==date].drop(columns='Zeitstempel')
        
        data = data.merge(self.metadata, on='SDO_ID')
        
        station_coords = data[['X', 'Y']].to_numpy()
        values = np.array(data['Wert'])
        return station_coords, values
    
class WindGetter_AT(WindGetter):
    _crs_in = pyproj.CRS("epsg:4326")
    
    def __init__(self, station_file, data_file, crs_out):
        self.station_file = station_file
        self.data_file = data_file
        self.crs_out = crs_out
        
        self._transformer = pyproj.transformer.Transformer.from_crs(self._crs_in, self.crs_out)
        self.metadata = self.get_metadata()
        return

    def _convert_coords(self, meta_row):
        lon = meta_row['Länge [°E]']
        lat = meta_row['Breite [°N]']
        meta_row['X'], meta_row['Y'] = self._transformer.transform(lat, lon)
        return meta_row
    
    def get_metadata(self):
        metadata = pd.read_csv(self.station_file, usecols=['id', 'Länge [°E]', 'Breite [°N]'])
        metadata['X'] = np.nan
        metadata['Y'] = np.nan
        metadata = metadata.apply(self._convert_coords, axis=1)
        
        metadata = metadata[['id', 'X', 'Y']]
        return metadata
    
    def get_measurements(self, date):
        data = pd.read_csv(self.data_file)
        data['time'] = pd.to_datetime(data['time'])
        data = data.rename(columns={'station': 'id'})
        
        pddate = pd.to_datetime(date + 'T00:00+00:00')
        data = data[(data['time']==pddate) & (data['vvmax'].notna())]
        
        data = data.merge(self.metadata, on='id')
        
        station_coords = data[['X', 'Y']].to_numpy()
        values = np.array(data['vvmax'])
        return station_coords, values

class WindGetter_CZ(WindGetter):
    _crs_in = pyproj.CRS("epsg:4326")
    
    def __init__(self, files, crs_out):
        self.files = files
        self.crs_out = crs_out
        
        self._transformer = pyproj.transformer.Transformer.from_crs(self._crs_in, self.crs_out)
        self.metadata = self.get_metadata()
        return
    
    def _convert_coords(self, meta_row):
        lon = float(meta_row['lon'])
        lat = float(meta_row['lat'])
        meta_row['X'], meta_row['Y'] = self._transformer.transform(lat, lon)
        return meta_row
    
    def get_metadata(self):
        metadata = []
        for file in self.files:
            with open(file) as f:
                lines = f.readlines()
                for start, line in enumerate(lines):
                    if 'METADATA' in line:
                        for stop, line in enumerate(lines):
                            if 'PØÍSTROJE' in line:
                                metadata += map(lambda x: x.replace(',','.').split(';'), lines[start+2:stop-1])
                                break
                        break

        metadata = pd.DataFrame(metadata, columns=['ID', 'name', 'start_date', 'end_date', 'lon', 'lat', 'alt'])
        metadata['start_date'] = pd.to_datetime(metadata['start_date'], format='%d.%m.%Y')
        metadata['end_date'] = pd.to_datetime(metadata['end_date'], format='%d.%m.%Y')
        metadata['lon'] = metadata['lon'].astype(float)
        metadata['lat'] = metadata['lat'].astype(float)
        
        metadata['X'] = np.nan
        metadata['Y'] = np.nan
        metadata = metadata.apply(self._convert_coords, axis=1)
        
        metadata = metadata[['ID', 'start_date', 'end_date', 'X', 'Y']]
        return metadata
    
    def get_measurements(self, date):
        data = []
        for file in self.files:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    if date.replace('-', ';') in line:
                        speed = float(line.split(';')[3].replace(',','.'))
                        
                        pddate = pd.to_datetime(date, format='%Y-%m-%d')
                        ID = file.split('/')[-1].split('\\')[-1].split('_')[0]
                        station = self.metadata[(self.metadata['ID'] == ID) & (self.metadata['start_date'] <= pddate) & (self.metadata['end_date'] >= pddate)]
                        
                        if len(station) == 0:
                            raise RuntimeError(f'No station metadata found for station {ID} on {date}')
                        if len(station) > 1:
                            raise RuntimeError(f'{len(station)} sets of metadata found for station {ID} on {date} instead of one unique set')
                        
                        data.append([station['X'].values[0], station['Y'].values[0], speed])
                        break
        
        data = np.array(data)
        return data[:, :2], data[:, 2]