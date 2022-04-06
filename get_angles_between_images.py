import os
import glob
import random
import subprocess

import s2p
import rpcm
from utils import *

#
result_angles_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/angles.csv'
image_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/JAX_DATA/GEOTIFF'
image_wildcard = 'JAX_214_*_PAN.tif'

#
result_angles_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/angles.csv'
image_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/OMA_DATA/GEOTIFF'
image_wildcard = 'OMA_203_*_PAN.tif'






image_filename_list = sorted( glob.glob(os.path.join(image_dir, image_wildcard)) )


# all the possible image pairs
all_pairs = get_all_possible_pairs_from_list(image_filename_list,order_matters=True)
pair_list = all_pairs
print('Pair list:', pair_list)

pair_angles_list = []

print('')
for i,pair in enumerate(pair_list):
    print('procesando {:03d}/{:03d}'.format(i, len(pair_list)), flush=True)
    ref_filename = pair[0]
    sec_filename = pair[1]
    _, ref_image_name = os.path.split(ref_filename)
    _, sec_image_name = os.path.split(sec_filename)
    ref_image_name_no_extension, _ = os.path.splitext(ref_image_name)
    sec_image_name_no_extension, _ = os.path.splitext(sec_image_name)

    # --------------------------------------------------------------------------
    # -----ANGLES---------------------------------------------------------------
    # angles [ref_zenith, ref_azimut, sec_zenith, sec_azimut, ref_sec_angle]
    # --------------------------------------------------------------------------
    angles = [ref_image_name_no_extension, sec_image_name_no_extension] + list(get_angles(ref_filename, sec_filename))
    pair_angles_list.append(angles)



with open(result_angles_filename, 'w') as f:
    f.write('ref sec ref_zenith ref_azimut sec_zenith sec_azimut ref_sec_angle\n')
    for item in pair_angles_list:
        f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))
