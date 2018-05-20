# Satellite image recognition using U-net CNN.

Project aim is to provide free web-application for geo-reseachers to find and extract usefull insights from map.

# Core 

Project consists of several modules

* Django web-application, which will be used to provide easy access to CNNs functionality, as well as interactive leaflet map;
* U-net (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) CNNs, trained to find roads, houses and cars (using Keras + Tensorflow);
* Leaflet map + sentinel (https://scihub.copernicus.eu/) tile layer: sentinel will be used as input for CNNs;
* OpenStreetMap module to load polygons: they will be used as a teacher for CNNs.

# Algorithm

1. Cover leaflet map with sentinel data and openstreetmap polygons;
2. Train several u-net CNNs to find roads, houses e.t.c;
3. Now researcher can choose area of interest, use one of pre-trained CNNs to extract the data he needs and download objects as .png or .wkt files.

# Usage

As of yet project is not fully complete, but you can use modules separetly.

1. Pre-trained u-net models are available at /satellite_image_recognition/models/*.hdf5. Input - 512 * 512 images in grey-scale format.
2. Web-application:
  2.1 Upload module (available at localhost:8000/upload/) - lets you upload satellite images and choose what objects to find on them;
  2.2 Map module - leaflet map with sentinel data (moscow only) and predicted roads polygons for that region.
3. Various helpful script:
  3.1 data.ipynb - prepare data for u-net;
  3.2 unet.ipynb - train / test u-net (can run on multiple gpus);
  3.3 combine_sentinel_and_osm.ipynb - helpful script to load osm polygons by lat, lon bounds.


