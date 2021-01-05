import os
import glob
import random
import subprocess

import s2p
import rpcm
from utils import *

random.seed(1234)
pair_count = 1000  #ALL PAIRS
dsm_resolution = 0.3
dsm_radius = 1

s2_config_template_filename = '//s2p_config_template.json'
vissat_colmap_config_template_filename = '//vissat_colmap_config_template.json'

s2p_ganet_main = '/home/agomez/ownCloud/Documents/doctorado/MultiViewStereo/python/methods_comparison/modified_s2p_for_ganet.py'
vissat_colmap_main = '/home/agomez/Software/MultiStereo/COLMAP/VisSatSatelliteStereo/stereo_pipeline.py'



crop_id = 'crop_01'

image_set_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped/mvs3d_gt_{}'.format(crop_id)
gt_filename = '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/ground_truth/mvs3d_gt_{}.tif'.format(crop_id)
image_list_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped/mvs3d_gt_{}_image_list.txt'.format(crop_id)



# JAX
image_sets = [156, 165, 214, 251, 264] # JAX
prefix = 'JAX'
#OMA
image_sets = [203, 247, 251, 287, 353]  #OMA
prefix = 'OMA'


# NEAR IN TIME
image_set = image_sets[4]
image_set_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/cropped/{0}_{1:03d}'.format(prefix, image_set)
gt_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/gt/{0}_{1:03d}_DSM.tif'.format(prefix, image_set)
output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_{}'.format(prefix)

# FAR IN TIME
image_set = image_sets[4]
image_set_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/cropped_far_in_time/{0}_{1:03d}'.format(prefix, image_set)
gt_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/gt/{0}_{1:03d}_DSM.tif'.format(prefix, image_set)
output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_FAR_IN_TIME{}'.format(prefix)



image_list_filename = os.path.join(image_set_dir, 'image_list.txt')

#
# image_set = 287   #203, 247, 251, 269, 287
# prefix = 'OMA'
# image_set_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/cropped/{0}_{1:03d}'.format(prefix, image_set)
# image_set_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/cropped_23_28/{0}_{1:03d}'.format(prefix, image_set)

# gt_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/{0}_DATA/gt/{0}_{1:03d}_DSM.tif'.format(prefix, image_set)
#image_list_filename = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/IARPA_DATA/cropped/{}_gt_{}_image_list.txt'.format(prefix, image_set)

#
#
#
# output_dir = '/home/agomez/Documents/iie/satelite/DATA/COMPARISON_RESULTS'
# output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON'
# output_dir = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA/COMPARISON_OMA'


#-------------------------------------------------------------------------------------------------------------

_, image_set_name = os.path.split(image_set_dir)

vissat_colmap_running_script_list=[]
vissat_colmap_dsm_result_list = []

# get image filenames in the set
image_filename_list = sorted(glob.glob(os.path.join(image_set_dir,'*.tif')))
if not os.path.isfile(image_list_filename):
    with open(image_list_filename, 'w') as f:
        for item in image_filename_list:
            f.write("%s\n" % item)


vissat_colmap_output_image_set_dir = os.path.join(output_dir, 'vissat_colmap', image_set_name)
if not os.path.isdir(vissat_colmap_output_image_set_dir):
    os.makedirs(vissat_colmap_output_image_set_dir)

#--------------------------------------------------------------------------
# -----VISSAT COLMAP-------------------------------------------------------
# --------------------------------------------------------------------------
config_filename = os.path.join(vissat_colmap_output_image_set_dir, 'multi_vissat_colmap_config.json')
work_dir = os.path.join(vissat_colmap_output_image_set_dir, 'multi')

# create the command to create the vissat_colmap config
create_vissat_colmap_configuration_command = 'python create_colmap_configuration.py'
create_vissat_colmap_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
create_vissat_colmap_configuration_command += ' --image_list_filename "{}"'
create_vissat_colmap_configuration_command += ' --gt_image_filename "{}" --work_dir "{}"'

create_vissat_colmap_configuration_command = create_vissat_colmap_configuration_command.format(vissat_colmap_config_template_filename,
                                                                           config_filename,
                                                                           image_list_filename,
                                                                           gt_filename, work_dir)
# create the configuration
subprocess.call(create_vissat_colmap_configuration_command, shell=True)

# script to run vissat colmap
vissat_colmap_running_script_list.append('python {} --config_file {}'.format(vissat_colmap_main, config_filename))

# where to find the dsm
#<work_dir>/colmap/mvs/dsm/dsm_tif/0000_<ref_image_name>.tif
vissat_colmap_dsm_result_list.append( '{}'.format( os.path.join(work_dir, 'mvs_results/aggregate_2p5d/aggregate_2p5d_dsm.tif')))

with open(os.path.join(vissat_colmap_output_image_set_dir, 'multi_running_script.sh'), 'w') as f:
    for item in vissat_colmap_running_script_list:
        f.write("%s\n" % item)

with open(os.path.join(vissat_colmap_output_image_set_dir, 'multi_dsm_result_filenames.txt'), 'w') as f:
    for item in vissat_colmap_dsm_result_list:
        f.write("%s\n" % item)


exit(0)


# --config_template_filename
# vissat_colmap_config_template.json
# --config_filename
# test_vissat_colmap/config_vissat_colmap.json
# --image_list_filename
# test_vissat_colmap/image_list.txt
# --gt_image_filename
# /home/agomez/ownCloud/Documents/doctorado/MultiViewStereo/data/JAX/JAX_068_DSM.tif
# --work_dir
# ./test_vissat_colmap/workdir




# all the possible image pairs
all_pairs = get_all_possible_pairs_from_list(image_filename_list)
#print('Pairs:  possible={}, selected={}'.format(len(all_pairs), pair_count))

# select some pairs (random or by angle criteria)
#pair_list = random.sample(all_pairs, pair_count)
pair_list = all_pairs
print('Pair list:', pair_list)


#create the output dir
s2p_output_image_set_dir = os.path.join(output_dir, 's2p', image_set_name)
if not os.path.isdir(s2p_output_image_set_dir):
    os.makedirs(s2p_output_image_set_dir)

s2p_ganet_output_image_set_dir = os.path.join(output_dir, 's2p_ganet', image_set_name)
if not os.path.isdir(s2p_ganet_output_image_set_dir):
    os.makedirs(s2p_ganet_output_image_set_dir)






s2p_running_script_list = []
s2p_ganet_running_script_list = []
vissat_colmap_running_script_list = []

s2p_dsm_result_list = []
s2p_ganet_dsm_result_list = []
vissat_colmap_dsm_result_list = []

pair_angles_list = []


for pair in pair_list:
    ref_filename = pair[0]
    sec_filename = pair[1]
    _, ref_image_name = os.path.split(ref_filename)
    _, sec_image_name = os.path.split(sec_filename)
    ref_image_name_no_extension, _ = os.path.splitext(ref_image_name)
    sec_image_name_no_extension, _ = os.path.splitext(sec_image_name)

    # --------------------------------------------------------------------------
    #-----S2P-------------------------------------------------------------------
    # --------------------------------------------------------------------------
    config_filename = os.path.join(s2p_output_image_set_dir, '{}_{}_s2p_config.json'.format(ref_image_name_no_extension, sec_image_name_no_extension))
    s2p_output_dir = os.path.join(s2p_output_image_set_dir, '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

    # create the command to create the s2p config
    create_s2p_configuration_command  = 'python create_s2p_configuration.py'
    create_s2p_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
    create_s2p_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
    create_s2p_configuration_command += ' --gt_image_filename "{}" --out_dir "{}"'
    create_s2p_configuration_command += ' --dsm_resolution {}'
    create_s2p_configuration_command += ' --dsm_radius {}'

    create_s2p_configuration_command = create_s2p_configuration_command.format(s2_config_template_filename,
                                                                               config_filename,
                                                                               ref_filename, sec_filename,
                                                                               gt_filename, s2p_output_dir,
                                                                               dsm_resolution, dsm_radius)
    # create the configuration
    subprocess.call(create_s2p_configuration_command, shell=True)

    # script to run s2p
    s2p_running_script_list.append('s2p {}'.format(config_filename))

    # where to find the dsm
    s2p_dsm_result_list.append('{}'.format( os.path.join(s2p_output_dir, 'dsm.tif')) )

    #angles [ref_zenith, ref_azimut, sec_zenith, sec_azimut, ref_sec_angle]
    angles = [ref_image_name_no_extension, sec_image_name_no_extension] + list(get_angles(ref_filename, sec_filename))
    pair_angles_list.append( angles )

    # --------------------------------------------------------------------------
    # -----S2P-GANET------------------------------------------------------------
    # --------------------------------------------------------------------------
    config_filename = os.path.join(s2p_ganet_output_image_set_dir, '{}_{}_s2p_config.json'.format(ref_image_name_no_extension,
                                                                                            sec_image_name_no_extension))
    s2p_ganet_output_dir = os.path.join(s2p_ganet_output_image_set_dir,
                                  '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

    # create the command to create the s2p config
    create_s2p_configuration_command = 'python create_s2p_configuration.py'
    create_s2p_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
    create_s2p_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
    create_s2p_configuration_command += ' --gt_image_filename "{}" --out_dir "{}"'
    create_s2p_configuration_command += ' --dsm_resolution {}'
    create_s2p_configuration_command += ' --dsm_radius {}'

    create_s2p_configuration_command = create_s2p_configuration_command.format(s2_config_template_filename,
                                                                               config_filename,
                                                                               ref_filename, sec_filename,
                                                                               gt_filename, s2p_ganet_output_dir,
                                                                               dsm_resolution, dsm_radius)
    # create the configuration
    subprocess.call(create_s2p_configuration_command, shell=True)

    # script to run s2p
    s2p_ganet_running_script_list.append('python {} {}'.format(s2p_ganet_main, config_filename))

    # where to find the dsm
    s2p_ganet_dsm_result_list.append('{}'.format(os.path.join(s2p_ganet_output_dir, 'dsm.tif')))



    #--------------------------------------------------------------------------
    # -----VISSAT COLMAP-------------------------------------------------------
    # --------------------------------------------------------------------------
    config_filename = os.path.join(vissat_colmap_output_image_set_dir, '{}_{}_vissat_colmap_config.json'.format(ref_image_name_no_extension,
                                                                                        sec_image_name_no_extension))
    work_dir = os.path.join(vissat_colmap_output_image_set_dir,
                                  '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

    # create the command to create the s2p config
    create_vissat_colmap_configuration_command = 'python create_colmap_configuration.py'
    create_vissat_colmap_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
    create_vissat_colmap_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
    create_vissat_colmap_configuration_command += ' --gt_image_filename "{}" --work_dir "{}"'

    create_vissat_colmap_configuration_command = create_vissat_colmap_configuration_command.format(vissat_colmap_config_template_filename,
                                                                               config_filename,
                                                                               ref_filename, sec_filename,
                                                                               gt_filename, work_dir)
    # create the configuration
    subprocess.call(create_vissat_colmap_configuration_command, shell=True)

    # script to run vissat colmap
    vissat_colmap_running_script_list.append('python {} --config_file {}'.format(vissat_colmap_main, config_filename))

    # where to find the dsm
    #<work_dir>/colmap/mvs/dsm/dsm_tif/0000_<ref_image_name>.tif
    vissat_colmap_dsm_result_list.append( '{}'.format( os.path.join(work_dir, 'colmap/mvs/dsm/dsm_tif/0000_{}.tif'.format(ref_image_name_no_extension)) ) )


# save the running scripts
with open(os.path.join(s2p_output_image_set_dir, 'running_script.sh'), 'w') as f:
    for item in s2p_running_script_list:
        f.write("%s\n" % item)
with open(os.path.join(s2p_ganet_output_image_set_dir, 'running_script.sh'), 'w') as f:
    for item in s2p_ganet_running_script_list:
        f.write("%s\n" % item)
with open(os.path.join(vissat_colmap_output_image_set_dir, 'running_script.sh'), 'w') as f:
    for item in vissat_colmap_running_script_list:
        f.write("%s\n" % item)

# save the results list
with open(os.path.join(s2p_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
    for item in s2p_dsm_result_list:
        f.write("%s\n" % item)
with open(os.path.join(s2p_ganet_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
    for item in s2p_ganet_dsm_result_list:
        f.write("%s\n" % item)
with open(os.path.join(vissat_colmap_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
    for item in vissat_colmap_dsm_result_list:
        f.write("%s\n" % item)

# save the angles list
with open(os.path.join(s2p_output_image_set_dir, 'angles.txt'), 'w') as f:
    for item in pair_angles_list:
        f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))

with open(os.path.join(s2p_ganet_output_image_set_dir, 'angles.txt'), 'w') as f:
    for item in pair_angles_list:
        f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))

with open(os.path.join(vissat_colmap_output_image_set_dir, 'angles.txt'), 'w') as f:
    for item in pair_angles_list:
        f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))


#
#
#
#
# python
# create_s2p_configuration.py - -config_template_filename. / s2p_config_template.json - -config_filename crops_s2p_config.json - -ref_image_filename TMP_CROP_KK_2.tif - -sec_image_filename TMP_CROP_KK_4.tif - -gt_image_filename
# '/home/agomez/VirtualBox VMs/OSGEOLive13/shared_folder/gt_crop_01.tif' - -out_dir. / crops_s2p - -dsm_resolution
# 0.3
#
#
# output_dir = '/home/agomez/Documents/iie/satelite/DATA/IARPA_DATA/cropped'
#
#
# for mvs3d_geotiff_filename in mvs3d_geotiff_filename_list:
#     image_filename = mvs3d_geotiff_filename
#     rpc = rpcm.rpc_from_geotiff(image_filename)
#
#     _, image_name = os.path.split(image_filename)
#     image_acquisition_date_string = image_name[16:29]
#
#     for mvs3d_gt_crop_filename in mvs3d_gt_crop_filename_list:
#         gt_filename = mvs3d_gt_crop_filename
#         _, gt_name = os.path.split(gt_filename)
#         gt_name_no_extension,_ = os.path.splitext(gt_name)
#
#         gt_cropped_data_dir  = os.path.join(output_dir, gt_name_no_extension)
#         if not os.path.isdir(gt_cropped_data_dir):
#             os.makedirs(gt_cropped_data_dir)
#
#         # get aoi from gt
#         aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
#             ref_filename=mvs3d_geotiff_filename,
#             gt_filename=gt_filename,
#             padding=aoi_relative_padding)
#
#
#
#         crop_image_filename = os.path.join(gt_cropped_data_dir, image_acquisition_date_string + '.tif')
#
#         x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=0)
#         s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_image_filename)
#
#
