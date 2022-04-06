import os
import glob
import argparse
import random
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.transform import EuclideanTransform, warp
from collections import namedtuple
import shutil
import modify_rectification



def get_rl_chip_filename(lr_chip_filename):
    lr_name = os.path.basename(lr_chip_filename)[:-len('_Rectified.tif')]
    # print(lr_name)
    [left_name, right_name] = lr_name.split('_and_')
    rl_name = right_name + '_and_' + left_name
    # print(rl_name)
    rl_filename = os.path.join(chips_directory, rl_name + '_Rectified.tif')
    return rl_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Re-rectifies the RVL dataset')
    parser.add_argument('rvl_base_directory', type=str)
    parser.add_argument('rvl_rerectified_base_directory', type=str)
    parser.add_argument('rvl_dataset_name', type=str, help='One of: Explorer, JAX, MP1, MP2, MP3, PS1, PS2, PS3')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    original_dataset_directory = os.path.join(args.rvl_base_directory, args.rvl_dataset_name)
    rerectified_dataset_directory  = os.path.join(args.rvl_rerectified_base_directory, args.rvl_dataset_name)

    if os.path.exists(rerectified_dataset_directory) and not args.overwrite:
        raise ValueError('Directotio de salida ya existe: {}'.format(rerectified_dataset_directory))

    chips_directory = os.path.join(original_dataset_directory, 'Rectified_Chips')
    disparity_directory = os.path.join(original_dataset_directory, 'Disparity')
    masks_directory = os.path.join(original_dataset_directory, 'Masks')
    metadata_directory = os.path.join(original_dataset_directory, 'Metadata')

    rerectified_chips_directory = os.path.join(rerectified_dataset_directory, 'Rectified_Chips')
    rerectified_disparity_directory = os.path.join(rerectified_dataset_directory, 'Disparity')
    rerectified_masks_directory = os.path.join(rerectified_dataset_directory, 'Masks')
    rerectified_metadata_directory = os.path.join(rerectified_dataset_directory, 'Metadata')

    os.makedirs(rerectified_chips_directory, exist_ok=True)
    os.makedirs(rerectified_disparity_directory, exist_ok=True)
    os.makedirs(rerectified_masks_directory, exist_ok=True)
    os.makedirs(rerectified_metadata_directory, exist_ok=True)


    chips_all_filenames = sorted(glob.glob(os.path.join(chips_directory, '*_Rectified.tif')))

    for lr_chip_filename in chips_all_filenames:
        rl_chip_filename = get_rl_chip_filename(lr_chip_filename)
        disparity_filename = lr_chip_filename.replace('Rectified_Chips', 'Disparity').replace('_Rectified.tif',
                                                                                              '_disp.tif')
        mask_filename = lr_chip_filename.replace('Rectified_Chips', 'Masks').replace('_Rectified.tif', '_mask.png')
        metadata_filename = lr_chip_filename.replace('Rectified_Chips', 'Metadata').replace('_Rectified.tif',
                                                                                            '_metadata.json')

        output_lr_chip_filename = lr_chip_filename.replace(chips_directory, rerectified_chips_directory)
        output_rl_chip_filename = rl_chip_filename.replace(chips_directory, rerectified_chips_directory)
        output_disparity_filename = lr_chip_filename.replace(chips_directory, rerectified_disparity_directory)
        output_mask_filename = lr_chip_filename.replace(chips_directory, rerectified_masks_directory)
        output_metadata_filename = lr_chip_filename.replace(chips_directory, rerectified_metadata_directory)

        modify_rectification.modify_rectification(lr_chip_filename, rl_chip_filename, disparity_filename, mask_filename,
                             output_lr_chip_filename, output_rl_chip_filename, output_disparity_filename, output_mask_filename,
                             precomputed_matches_filename=None,
                             register_horizontally_shear=True,
                             register_horizontally_translation=True,
                             register_horizontally_translation_flag='negative',
                             debug=False)

        shutil.copy2(metadata_filename, output_metadata_filename)