import os
import sys

import numpy as np
import s2p
import rpcm
from utils import *
import pandas as pd



aoi_relative_padding = 0.15


# image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/GEOTIFF'
# gt_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/gt'
# cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped'
#
# # old set of near in time images
# image_sets=[22, 33, 68, 72, 79]
# image_numbers = [[4,5,6,11,12,18],
#                  [4,5,6,7,11,15],
#                  [4,5,6,7,11,18],
#                  [4,5,6,7,11,18],
#                  [4,5,6,7,11,15]
#                  ]
# # new set of near in time images
# cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped'
# image_sets=[156, 165, 214, 251, 264]
# image_numbers = [[6,7,8,9,10,11],
#                     [6,7,8,9,10,11],
#                     [6,7,8,9,10,11],
#                     [6,7,8,9,10,11],
#                     [6,7,8,9,10,11],
#                  ]
#
# # new set of far in time images
# cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped_far_in_time'
# image_sets=[156, 165, 214, 251, 264]
# image_numbers = [[3, 7, 12, 13, 20, 22],
#                 [3, 7, 12, 13, 20, 22],
#                 [3, 7, 12, 13, 20, 22],
#                 [3, 7, 12, 13, 20, 22],
#                 [3, 7, 12, 13, 20, 22],
#                  ]
#
# # new set of all images
# cropped_image_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/cropped_all'
# image_sets=[156, 165, 214, 251, 264]
# image_numbers = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  18,19,20,   22,23,  25],
#                  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  18,19,20,21,22,23,  25],
#                  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  18,19,20,21,22,23,  25,26],
#                  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  18,19,20,21,22,23,  25,26],
#                  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,  18,19,20,21,22,23,  25,26],
#                  ]
#
#
# # SET THE INDEX TO RUN---------------
# i=4
# #------------------------------------
# image_set = image_sets[i]
#
# cropped_image_dir = os.path.join(cropped_image_base_dir, 'JAX_{:03d}'.format(image_set))
# if not os.path.isdir(cropped_image_dir):
#     os.makedirs(cropped_image_dir)
#
#
# gt_filename = os.path.join(gt_base_dir,'JAX_{:03d}_DSM.tif'.format(image_sets[i]))
# _, gt_name = os.path.split(gt_filename)
# gt_name_no_extension,_ = os.path.splitext(gt_name)
#
# for n in image_numbers[i]: #image_filename in image_filenames:
#     image_filename = os.path.join(image_base_dir, 'JAX_{:03d}_{:03d}_PAN.tif'.format(image_set, n))
#
#     rpc = rpcm.rpc_from_geotiff(image_filename)
#
#     _, image_name = os.path.split(image_filename)
#     #image_acquisition_date_string = image_name[16:29]
#
#     # get aoi from gt
#     aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
#         ref_filename=image_filename,
#         gt_filename=gt_filename,
#         padding=aoi_relative_padding)
#
#
#
#     crop_image_filename = os.path.join(cropped_image_base_dir, 'JAX_{:03d}'.format(image_set), 'JAX_{:03d}_{:03d}_PAN_CROPPED.tif'.format(image_set,n ) )
#
#     x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=0)
#     s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)


if __name__ == '__main__':
    # from skimage.io import imread, imsave
    import argparse

    parser = argparse.ArgumentParser(description='Generate JAX or OMA cropped images')
    parser.add_argument('cropped_image_base_dir', type=str)
    parser.add_argument('--dataset', type=str, default='JAX', help='JAX or OMA')
    parser.add_argument('--image_set_count', type=int, default=1)
    parser.add_argument('--image_count', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--web_list_csv_filename', type=str, default='/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/jax_oma_listado_web.csv')

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    web_list_df = pd.read_csv(args.web_list_csv_filename)
    # print(web_list_df)

    #restrict to the dataset
    df = web_list_df.loc[web_list_df['dataset']=='JAX']

    # get the available image sets (regions) and the number of images in those image sets
    available_image_sets = np.unique(df['image_set'].values)
    available_image_sets_image_count = df['image_set'].value_counts().sort_index().values
    print(available_image_sets)
    print(available_image_sets_image_count)

    # keep the image sets that have at least args.image_count images
    valid_indices = available_image_sets_image_count > args.image_count
    valid_available_image_sets = available_image_sets[valid_indices]
    valid_available_image_sets_image_count = available_image_sets_image_count[valid_indices]
    print(valid_available_image_sets)
    print(valid_available_image_sets_image_count)

    # sample the image sets
    image_sets = np.random.choice(valid_available_image_sets, args.image_set_count, replace=False)
    print(image_sets)

    # sample the images
    for i,image_set in enumerate(image_sets):
        available_image_numbers = df.loc[df['image_set']==image_set,'image_number'].values
        image_numbers = np.random.choice(available_image_numbers, args.image_count, replace=False)
        print(image_set)
        print(available_image_numbers)
        print(image_numbers)

        cropped_image_dir = os.path.join(args.cropped_image_base_dir, '{}_{:03d}'.format(args.dataset, image_set))
        if not os.path.isdir(cropped_image_dir):
            os.makedirs(cropped_image_dir)

        gt_base_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/gt_all'
        gt_filename = os.path.join(gt_base_dir,'JAX_{:03d}_DSM.tif'.format(image_sets[i]))
        _, gt_name = os.path.split(gt_filename)
        gt_name_no_extension,_ = os.path.splitext(gt_name)

        for n in image_numbers: #image_filename in image_filenames:
            image_filename = '/vsicurl/{}'.format(df.loc[(df['image_set']==image_set) & (df['image_number']==n), 'url'].item())

            rpc = rpcm.rpc_from_geotiff(image_filename)

            _, image_name = os.path.split(image_filename)
            #image_acquisition_date_string = image_name[16:29]

            print(rpc)
            print(image_name)

            # get aoi from gt
            aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
                ref_filename=image_filename,
                gt_filename=gt_filename,
                padding=aoi_relative_padding)



            crop_image_filename = os.path.join(cropped_image_dir, '{}_{:03d}_{:03d}_PAN_CROPPED.tif'.format(args.dataset,image_set,n ) )

            x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=0)
            s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)
