import os
import sys

import s2p
import rpcm
from utils import *

aoi_relative_padding = 0.15

image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/GEOTIFF'
gt_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/gt'

# near in time images
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/cropped'
image_sets=[203, 247, 251, 287, 353]
image_numbers = [23,24,25,26,27,28]

# far in time images
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/cropped_far_in_time'
image_sets=[203, 247, 251, 287, 353]
image_numbers = [1,9,15,20,32,41]

# new set of all images
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/cropped_all'
image_sets=[203, 247, 251, 287, 353]
image_numbers = [[t for t in range(1,44) if t not in []],
                 [t for t in range(1,44) if t not in [7,10,12,42]],
                 [t for t in range(1,44) if t not in [35]],
                 [t for t in range(1,44) if t not in [13]],
                 [t for t in range(1,44) if t not in [43]],

                 ]



# SET THE INDEX TO RUN---------------
i=4
#------------------------------------
image_set = image_sets[i]

cropped_image_dir = os.path.join(cropped_image_base_dir, 'OMA_{:03d}'.format(image_set))
if not os.path.isdir(cropped_image_dir):
    os.makedirs(cropped_image_dir)


gt_filename = os.path.join(gt_base_dir,'OMA_{:03d}_DSM.tif'.format(image_sets[i]))
_, gt_name = os.path.split(gt_filename)
gt_name_no_extension,_ = os.path.splitext(gt_name)

for n in image_numbers[i]: #image_filename in image_filenames:
    image_filename = os.path.join(image_base_dir, 'OMA_{:03d}_{:03d}_PAN.tif'.format(image_set, n))

    rpc = rpcm.rpc_from_geotiff(image_filename)

    _, image_name = os.path.split(image_filename)
    #image_acquisition_date_string = image_name[16:29]

    # get aoi from gt
    aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
        ref_filename=image_filename,
        gt_filename=gt_filename,
        padding=aoi_relative_padding)



    crop_image_filename = os.path.join(cropped_image_base_dir, 'OMA_{:03d}'.format(image_set), 'OMA_{:03d}_{:03d}_PAN_CROPPED.tif'.format(image_set,n ) )

    x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=min_height)
    s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)


