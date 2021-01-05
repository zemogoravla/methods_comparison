import os
import sys

import s2p
import rpcm
from utils import *

aoi_relative_padding = 0.15

image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/GEOTIFF'
gt_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/gt'
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped'

# old set of near in time images
image_sets=[22, 33, 68, 72, 79]
image_numbers = [[4,5,6,11,12,18],
                 [4,5,6,7,11,15],
                 [4,5,6,7,11,18],
                 [4,5,6,7,11,18],
                 [4,5,6,7,11,15]
                 ]
# new set of near in time images
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped'
image_sets=[156, 165, 214, 251, 264]
image_numbers = [[6,7,8,9,10,11],
                    [6,7,8,9,10,11],
                    [6,7,8,9,10,11],
                    [6,7,8,9,10,11],
                    [6,7,8,9,10,11],
                 ]

# new set of far in time images
cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped_far_in_time'
image_sets=[156, 165, 214, 251, 264]
image_numbers = [[3, 7, 12, 13, 20, 22],
                [3, 7, 12, 13, 20, 22],
                [3, 7, 12, 13, 20, 22],
                [3, 7, 12, 13, 20, 22],
                [3, 7, 12, 13, 20, 22],
                 ]



# SET THE INDEX TO RUN---------------
i=4
#------------------------------------
image_set = image_sets[i]

cropped_image_dir = os.path.join(cropped_image_base_dir, 'JAX_{:03d}'.format(image_set))
if not os.path.isdir(cropped_image_dir):
    os.makedirs(cropped_image_dir)


gt_filename = os.path.join(gt_base_dir,'JAX_{:03d}_DSM.tif'.format(image_sets[i]))
_, gt_name = os.path.split(gt_filename)
gt_name_no_extension,_ = os.path.splitext(gt_name)

for n in image_numbers[i]: #image_filename in image_filenames:
    image_filename = os.path.join(image_base_dir, 'JAX_{:03d}_{:03d}_PAN.tif'.format(image_set, n))

    rpc = rpcm.rpc_from_geotiff(image_filename)

    _, image_name = os.path.split(image_filename)
    #image_acquisition_date_string = image_name[16:29]

    # get aoi from gt
    aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
        ref_filename=image_filename,
        gt_filename=gt_filename,
        padding=aoi_relative_padding)



    crop_image_filename = os.path.join(cropped_image_base_dir, 'JAX_{:03d}'.format(image_set), 'JAX_{:03d}_{:03d}_PAN_CROPPED.tif'.format(image_set,n ) )

    x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=0)
    s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)


