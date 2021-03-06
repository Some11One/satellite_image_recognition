import io
import json
import os
import pickle
import zipfile

import geojson
import geopandas as gpd
import numpy as np
from PIL import Image
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render
from geojson import Feature, FeatureCollection
from geopandas_osm import osm
from keras.preprocessing.image import img_to_array, array_to_img
from shapely.geometry import shape

from . import u_net
from .settings import BASE_DIR


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def getfiles(request, filenames):
    zip_subdir = "result"
    zip_filename = "%s.zip" % zip_subdir

    # Open StringIO to grab in-memory ZIP contents
    s = io.BytesIO()

    # The zip compressor
    zf = zipfile.ZipFile(s, "w")

    for fpath in filenames:
        try:
            # Calculate path for file in zip
            fdir, fname = os.path.split(fpath)
            zip_path = os.path.join(zip_subdir, fname)

            # Add file, at correct path
            zf.write(fpath, zip_path)

        except Exception as e:
            print(e)  # is ok for waterway prediction if none exists

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = HttpResponse(s.getvalue(), content_type="application/x-zip-compressed")
    # ..and correct content-disposition
    resp['Content-Disposition'] = 'attachment; filename=%s' % zip_filename

    return resp


d = load_obj(os.path.join(BASE_DIR, 'data/tile_dict'))


def upload(request):
    data = {}
    if "GET" == request.method:
        return render(request, "upload.html", data)
    try:
        photo_file = request.FILES["photo_file"]
        objects = request.POST.getlist('objects')
        if len(objects) == 0:
            messages.add_message(request, messages.ERROR, 'Выберите объекты для предсказания!')
            return render(request, "upload.html", data)
        if (not photo_file.name.endswith('.png')) and (not photo_file.name.endswith('.tif')):
            messages.add_message(request, messages.ERROR, 'Файл должен быть в .png или .tif формате!')
            return render(request, "upload.html", data)
        if photo_file.multiple_chunks():
            messages.add_message(request, messages.ERROR, 'Файл слишком большой!')
            return render(request, "upload.html", data)
        imgs = [photo_file]

        # create filename
        files = os.listdir(os.path.join(BASE_DIR, "satellite_image_recognition/media/"))
        max_image = '0'
        for file in files:
            if '.png' in file:
                max_image_temp = file.split('.')[0].split('_')[-1]
                if int(max_image_temp) > int(max_image):
                    max_image = max_image_temp
        max_image = str(int(max_image) + 1)
        data.update({'temp_image': "../media/temp_image_%s.png" % max_image})
        data.update({'temp_image_pred': "../media/temp_image_pred_%s.png" % max_image})

        # preprocess file - split into chunks
        imgdatas = np.ndarray((len(imgs) * 9, 512, 512, 1), dtype=np.uint8)
        i = 0
        for imgname in imgs:
            midname = imgname

            img = midname.read()
            img = Image.open(io.BytesIO(img))

            # save data
            img.save(os.path.join(BASE_DIR, 'satellite_image_recognition/media/temp_image_%s.png' % max_image))

            img = img_to_array(img)

            img_1 = img[0:512, 0:512, :]
            imgdatas[i] = img_1
            i += 1

            img_2 = img[0:512, 512:1024, :]
            imgdatas[i] = img_2
            i += 1

            img_3 = img[0:512, 988:1500, :]
            imgdatas[i] = img_3
            i += 1

            img_4 = img[512:1024, 0:512, :]
            imgdatas[i] = img_4
            i += 1

            img_5 = img[512:1024, 512:1024, :]
            imgdatas[i] = img_5
            i += 1

            img_6 = img[512:1024, 988:1500, :]
            imgdatas[i] = img_6
            i += 1

            img_7 = img[988:1500, 0:512, :]
            imgdatas[i] = img_7
            i += 1

            img_8 = img[988:1500, 512:1024, :]
            imgdatas[i] = img_8
            i += 1

            img_9 = img[988:1500, 988:1500, :]
            imgdatas[i] = img_9
            i += 1
        img_test = imgdatas.astype('float32')
        img_test /= 255
        res = np.zeros((9, 512, 512, 3))

        # predict
        for obj in objects:

            # u-net obj
            myunet = u_net.myUnet(tp=obj)
            i_obj = myunet.predict(img_test)
            i_obj[i_obj > 0.5] = 1
            i_obj[i_obj <= 0.5] = 0

            if obj == 'cars':
                res[:, :, :, 2:3] = i_obj[:, :, :, :1]
            elif obj == 'buildings':
                res[:, :, :, 1:2] = i_obj[:, :, :, :1]
        merged_res = np.zeros((1, 1500, 1500, 3))
        for j, i in enumerate(range(0, len(res), 9)):
            p1 = res[i]  # 0-512, 0-512
            merged_res[j, 0:512, 0:512, :] = p1

            p2 = res[i + 1]  # 0-512, 512-1024
            merged_res[j, 0:512, 512:1024, :] = p2

            p3 = res[i + 2]  # 0-512, 988-1500
            merged_res[j, 0:512, 988:1500, :] = p3

            p4 = res[i + 3]  # 512-1024, 0-512
            merged_res[j, 512:1024, 0:512, :] = p4

            p5 = res[i + 4]  # 512-1024, 512-1024
            merged_res[j, 512:1024, 512:1024, :] = p5

            p6 = res[i + 5]  # 512-1024, 988-1500
            merged_res[j, 512:1024, 988:1500, :] = p6

            p7 = res[i + 6]  # 988-1500, 0-512
            merged_res[j, 988:1500, 0:512, :] = p7

            p8 = res[i + 7]  # 988-1500, 512-1024
            merged_res[j, 988:1500, 512:1024, :] = p8

            p9 = res[i + 8]  # 988-1500, 988-1500
            merged_res[j, 988:1500, 988:1500, :] = p9

        # calc cars
        if 'cars' in objects:
            for index, photo in enumerate(merged_res):
                mtx = photo[:, :, 2]
                res = np.zeros((mtx.shape[0], mtx.shape[1]))

                res_count = 0
                for i in range(0, mtx.shape[0]):
                    for j in range(0, mtx.shape[1]):
                        if mtx[i, j] == 1:

                            if (i > 0 and mtx[i - 1, j] == 1):
                                res[i, j] = res[i - 1, j]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif (j > 0 and mtx[i, j - 1] == 1):
                                res[i, j] = res[i, j - 1]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif i < 1499 and mtx[i + 1, j] == 1:
                                res[i, j] = res[i + 1, j]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif j < 1499 and mtx[i, j + 1] == 1:
                                res[i, j] = res[i, j + 1]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            else:
                                res_count += 1
                                res[i, j] = res_count

                data.update({'cars_count': res_count})
        else:
            data.update({'cars_count': "-"})

        # calc houses
        if 'buildings' in objects:
            for index, photo in enumerate(merged_res):

                mtx = photo[:, :, 1]
                res = np.zeros((mtx.shape[0], mtx.shape[1]))

                res_count = 0
                for i in range(0, mtx.shape[0]):
                    for j in range(0, mtx.shape[1]):
                        if mtx[i, j] == 1:

                            if (i > 0 and mtx[i - 1, j] == 1):
                                res[i, j] = res[i - 1, j]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif (j > 0 and mtx[i, j - 1] == 1):
                                res[i, j] = res[i, j - 1]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif i < 1499 and mtx[i + 1, j] == 1:
                                res[i, j] = res[i + 1, j]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            elif j < 1499 and mtx[i, j + 1] == 1:
                                res[i, j] = res[i, j + 1]
                                if res[i, j] == 0:
                                    res_count += 1
                                    res[i, j] = res_count

                            else:
                                res_count += 1
                                res[i, j] = res_count

                data.update({'houses_count': int(res_count / 65)})
        else:
            data.update({'houses_count': "-"})

        array_to_img(merged_res[0]).save(
            os.path.join(BASE_DIR, 'satellite_image_recognition/media/temp_image_pred_%s.png' % max_image))

    except Exception as e:
        print(e)

    return render(request, "results.html", data)


def results(request):
    data = {}
    files = os.listdir(os.path.join(BASE_DIR, "satellite_image_recognition/media/"))
    max_image = '0'
    for file in files:
        if '.png' in file:
            max_image_temp = file.split('.')[0].split('_')[-1]
            if int(max_image_temp) > int(max_image):
                max_image = max_image_temp
    data.update({'temp_image': "../media/temp_image_%s.png" % max_image})
    data.update({'temp_image_pred': "../media/temp_image_pred_%s.png" % max_image})

    if request.GET.get('export'):
        image = Image.open(
            os.path.join(BASE_DIR, 'satellite_image_recognition/media/temp_image_pred_%s.png' % max_image))
        format = image.format
        extension = str(format)
        response = HttpResponse(content_type='image/' + extension.lower())
        response['Content-Disposition'] = 'attachment; filename=%s' % 'prediction.png'
        image.save(response, format)

        return response

    data.update({'cars_count': "-"})
    data.update({'houses_count': "-"})

    return render(request, 'results.html', data)


def map(request):
    latlng = request.GET.get('latlng', 'LatLng(55.751244, 37.618423)').strip('LatLng(').strip(')')
    base_lat = float(latlng.split(', ')[1])
    base_lon = float(latlng.split(', ')[0])
    r_lat = 0.01
    r_lon = 0.005

    o = {
        "coordinates": [[[base_lat - r_lat, base_lon - r_lon],
                         [base_lat + r_lat, base_lon - r_lon], [base_lat + r_lat, base_lon + r_lon],
                         [base_lat - r_lat, base_lon + r_lon], [base_lat - r_lat, base_lon - r_lon]]],
        "type": "Polygon"
    }

    s = json.dumps(o)
    g1 = geojson.loads(s)
    g2 = shape(g1)

    my_feature = Feature(geometry=g2)
    collection = FeatureCollection([my_feature])
    col = gpd.GeoDataFrame.from_features(collection['features'])

    try:
        df_highway = osm.query_osm('way', col.ix[0].geometry, recurse='down', tags='highway')
        df_highway = df_highway[['geometry']]
        df_highway['tag'] = 'highway'
        df_building = osm.query_osm('way', col.ix[0].geometry, recurse='down', tags='building')
        df_building = df_building[['geometry']]
        df_building['tag'] = 'building'
        df_waterway = osm.query_osm('way', col.ix[0].geometry, recurse='down', tags='waterway')
        df_waterway = df_waterway[['geometry']]
        df_waterway['tag'] = 'waterway'
        df_highway.append(df_building).append(df_waterway).to_csv(os.path.join(BASE_DIR, 'polygons.csv'), index=False)

        wkt = []
        test = [base_lon, base_lat]

        res = ()
        for item in d.items():
            k = item[0]
            v = item[1]
            tile_lat_min, tile_long_min, tile_lat_max, tile_long_max = v

            if tile_lat_min <= test[0] <= tile_lat_max and tile_long_min <= test[1] <= tile_long_max:
                res = (k, v)

        if len(res) > 0:
            x, y, z = res[0].split('_')

            return getfiles(request, [
                os.path.join(BASE_DIR, 'satellite_image_recognition/media/combined_37_UDB/%s/%s/%s.png' % (z, x, y)),
                os.path.join(BASE_DIR,
                             'satellite_image_recognition/media/combined_37_UDB/%s/%s/%s_pred_highway.png' % (z, x, y)),
                os.path.join(BASE_DIR,
                             'satellite_image_recognition/media/combined_37_UDB/%s/%s/%s_pred_waterway.png' % (
                             z, x, y)),
                os.path.join(BASE_DIR,
                             'satellite_image_recognition/media/combined_37_UDB/%s/%s/%s_pred_building.png' % (
                             z, x, y)),
                os.path.join(BASE_DIR, 'polygons.csv')])


    except Exception as e:
        print(e)
        wkt = []

    return render(request, 'map.html', {'base_lat': base_lon, 'base_lon': base_lat,
                                        'boundary': o, 'geometry': wkt})
