import os
import glob
import subprocess
import json
import sys
import datetime

import s2p
import rpcm
from utils import *

aoi_relative_padding = 0.1

base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/'
image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/GEOTIFF'
gt_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/gt'
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/cropped'

prefix = 'OMA'
image_wildcard = 'OMA_203_*_PAN.tif'

TMP_GDALINFO_JSON_FILENAME = '/home/agomez/tmp/tmp_gdalinfo.json'
RESULTS_FILENAME = os.path.join(base_dir, 'metadata.csv')

# tifs in the image_base_dir
image_filenames = sorted(glob.glob(os.path.join(image_base_dir, image_wildcard)))

results = []
results.append(['image', 'image_set', 'image_number', 'year', 'month', 'day', 'hour', 'minutes', 'seconds'])
for image_filename in image_filenames:
    _, image_name = os.path.split(image_filename)
    image_name_no_extension, _ = os.path.splitext(image_name)

    # run gdalinfo and get a json
    command = 'gdalinfo -json {} > {}'.format(image_filename, TMP_GDALINFO_JSON_FILENAME)
    subprocess.call(command, shell=True)
    # read the json
    with open(TMP_GDALINFO_JSON_FILENAME, 'r') as f:
        gdalinfo = json.load(f)

    date_time = gdalinfo['metadata']['']['NITF_IDATIM']

    image_set = image_name_no_extension[4:7]
    image_number = image_name_no_extension[8:11]
    year = int(date_time[:4])
    month = int(date_time[4:6])
    day = int(date_time[6:8])
    hour = int(date_time[8:10])
    minutes = int(date_time[10:12])
    seconds = int(date_time[12:14])

    results.append([image_name_no_extension, image_set, image_number, year, month, day, hour, minutes, seconds])

np.savetxt(RESULTS_FILENAME, results, fmt='%s')
