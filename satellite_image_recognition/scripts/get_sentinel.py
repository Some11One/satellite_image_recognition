from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

import json

import shapely.geometry
from cw_unet.scripts.geopandas_osm import osm
from shapely.geometry import shape

from shapely.wkt import loads

import pandas as pd
import json

import geojson
import geopandas as gpd
from geojson import Feature, Point, FeatureCollection

# connect to the API
api = SentinelAPI('korndmelentev', '21Korol.ev27', 'https://scihub.copernicus.eu/dhus')

# download single scene by known product id
# api.download(<product_id>)

# search by polygon, time, and Hub query keywords
# o = {
#     "coordinates": [[[23.314208, 37.768469], [24.039306, 37.768469], [24.039306, 38.214372], [23.314208, 38.214372],
#                      [23.314208, 37.768469]]],
#     "type": "Polygon"
# }
base_lat = 23.5304318
base_lon = 38.0572542

o = {
    "coordinates": [[[base_lat - 0.005, base_lon - 0.005],
                     [base_lat + 0.005, base_lon - 0.005], [base_lat + 0.005, base_lon + 0.005],
                     [base_lat - 0.005, base_lon + 0.005], [base_lat - 0.005, base_lon - 0.005]]],
    "type": "Polygon"
}
s = json.dumps(o)
g1 = geojson.loads(s)
g2 = shape(g1)
# footprint = geojson_to_wkt(read_geojson('map.geojson'))
footprint = g2.wkt
products = api.query(footprint,
                     date=('20151219', date(2015, 12, 29)),
                     platformname='Sentinel-2',
                     cloudcoverpercentage=(0, 30))

print(products)

# download all results from the search
# a = api.download_all(products)

# GeoJSON FeatureCollection containing footprints and metadata of the scenes
# b = api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
# c = api.to_geopandas(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
# api.get_product_odata(<product_id>)

# Get the product's full metadata available on the server
# api.get_product_odata(<product_id>, full=True)

# d = 7