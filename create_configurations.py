import os
import glob
import random
import subprocess

import s2p
import rpcm
from utils import *

random.seed(1234)
pair_count = 100000  #ALL PAIRS
dsm_radius = 1
tile_size_s2p = 600
tile_size_s2p_ganet = 1200 # quiero que sea una sola
disp_range_method_s2p_ganet = "fixed_altitude_range"
order_of_pairs_matter = True

DATA_BASE_DIRECTORY = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA_CLUSTER'
# para el cluster
#DATA_BASE_DIRECTORY = '/clusteruy/home/agomez/DATA'

CONFIGURATIONS = ['MVS_NIT', 'JAX_NIT', 'OMA_NIT', 'JAX_FIT', 'OMA_FIT', 'MVS_ALL', 'JAX_ALL']
CURRENT_CONFIGURATION = 'JAX_ALL'

configure_s2p = True
configure_s2p_ganet = False
configure_vissat_colmap = False
configure_vissat_planesweep = False

s2_config_template_filename = 's2p_config_template.json'
vissat_colmap_config_template_filename = 'vissat_colmap_config_template.json'

s2p_main = 's2p'
s2p_ganet_main = '/home/agomez/ownCloud/Documents/doctorado/MultiViewStereo/python/methods_comparison/modified_s2p_for_ganet.py'
vissat_colmap_main = '/home/agomez/Software/MultiStereo/COLMAP/VisSatSatelliteStereo/stereo_pipeline.py'
vissat_planesweep_exe = '/home/agomez/Software/MultiStereo/COLMAP/SatellitePlaneSweep/build/bin/satellitePlaneSweep'
plyflatten_exe = '/home/agomez/Software/satelite/s2p_ultimate/s2p/bin/plyflatten'

# para el cluster
#s2p_main = 'sbatch s2p_job.sh'
#s2p_ganet_main = 'sbatch s2p_ganet_job.sh'




if CURRENT_CONFIGURATION=='MVS_NIT':
    # MVS3D
    dsm_resolution = 0.3 #MVS3D
    image_sets = ['crop_01', 'crop_02', 'crop_03', 'crop_04', 'crop_05']
    prefix = 'MVS'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY, '{0}_DATA/cropped_nit/mvs3d_gt_{1}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/mvs3d_gt_{1}.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY,'COMPARISON_{}_NIT')

if CURRENT_CONFIGURATION=='JAX_NIT':
    #JAX NEAR IN TIME
    dsm_resolution = 0.5 #JAX, OMAHA
    image_sets = [156, 165, 214, 251, 264]
    prefix = 'JAX'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/cropped_nit/{0}_{1:03d}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/{0}_{1:03d}_DSM.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY,'COMPARISON_{}_NIT')

if CURRENT_CONFIGURATION == 'OMA_NIT':
    #OMA  NEAR IN TIME
    dsm_resolution = 0.5 #JAX, OMAHA
    image_sets = [203, 247, 251, 287, 353]
    prefix = 'OMA'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/cropped_nit/{0}_{1:03d}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/{0}_{1:03d}_DSM.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY,'COMPARISON_{}_NIT')

if CURRENT_CONFIGURATION == 'JAX_FIT':
    #JAX FAR IN TIME
    dsm_resolution = 0.5 #JAX, OMAHA
    image_sets = [156, 165, 214, 251, 264]
    prefix = 'JAX'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/cropped_fit/{0}_{1:03d}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/{0}_{1:03d}_DSM.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY, 'COMPARISON_{}_FIT')

if CURRENT_CONFIGURATION == 'OMA_FIT':
    #OMA FAR IN TIME
    dsm_resolution = 0.5 #JAX, OMAHA
    image_sets = [203, 247, 251, 287, 353]
    prefix = 'OMA'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/cropped_fit/{0}_{1:03d}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/{0}_{1:03d}_DSM.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY, 'COMPARISON_{}_FIT')

if CURRENT_CONFIGURATION=='MVS_ALL':
    # MVS3D
    dsm_resolution = 0.3 #MVS3D
    image_sets = ['crop_01', 'crop_02', 'crop_03', 'crop_04', 'crop_05']
    prefix = 'MVS'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY, '{0}_DATA/cropped_all/mvs3d_gt_{1}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/mvs3d_gt_{1}.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY,'COMPARISON_{}_ALL')

if CURRENT_CONFIGURATION=='JAX_ALL':
    #JAX NEAR IN TIME
    dsm_resolution = 0.5 #JAX, OMAHA
    image_sets = [156, 165, 214, 251, 264]
    prefix = 'JAX'
    image_set_dir_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/cropped_all/{0}_{1:03d}')
    gt_filename_template = os.path.join(DATA_BASE_DIRECTORY,'{0}_DATA/gt/{0}_{1:03d}_DSM.tif')
    output_dir_template = os.path.join(DATA_BASE_DIRECTORY,'COMPARISON_{}_ALL')

#-------------------------------------------------------------------------------------------------------------
for image_set in image_sets:
    image_set_dir = image_set_dir_template.format(prefix, image_set)
    gt_filename = gt_filename_template.format(prefix, image_set)
    output_dir = output_dir_template.format(prefix)



    _, image_set_name = os.path.split(image_set_dir)




    # get image filenames in the set
    image_filename_list = sorted(glob.glob(os.path.join(image_set_dir,'*.tif')))

    # all the possible image pairs
    all_pairs = get_all_possible_pairs_from_list(image_filename_list, order_matters=order_of_pairs_matter)
    #print('Pairs:  possible={}, selected={}'.format(len(all_pairs), pair_count))

    # select some pairs (random or by angle criteria)
    #pair_list = random.sample(all_pairs, pair_count)
    pair_list = all_pairs
    print('Pair list:', pair_list)


    #create the output dir
    s2p_output_image_set_dir = os.path.join(output_dir, 's2p', image_set_name)
    s2p_ganet_output_image_set_dir = os.path.join(output_dir, 's2p_ganet', image_set_name)
    vissat_colmap_output_image_set_dir = os.path.join(output_dir, 'vissat_colmap', image_set_name)
    vissat_planesweep_output_image_set_dir = os.path.join(output_dir, 'vissat_planesweep', image_set_name)

    if configure_s2p:
        if not os.path.isdir(s2p_output_image_set_dir):
            os.makedirs(s2p_output_image_set_dir)
    if configure_s2p_ganet:
        if not os.path.isdir(s2p_ganet_output_image_set_dir):
            os.makedirs(s2p_ganet_output_image_set_dir)

    if configure_vissat_colmap:
        if not os.path.isdir(vissat_colmap_output_image_set_dir):
            os.makedirs(vissat_colmap_output_image_set_dir)

    if configure_vissat_planesweep:
        if not os.path.isdir(vissat_planesweep_output_image_set_dir):
            os.makedirs(vissat_planesweep_output_image_set_dir)


    s2p_running_script_list = []
    s2p_ganet_running_script_list = []
    vissat_colmap_running_script_list = []
    vissat_planesweep_running_script_list = []

    s2p_dsm_result_list = []
    s2p_ganet_dsm_result_list = []
    vissat_colmap_dsm_result_list = []
    vissat_planesweep_dsm_result_list = []

    pair_angles_list = []

    for pair in pair_list:
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

        # --------------------------------------------------------------------------
        #-----S2P-------------------------------------------------------------------
        # --------------------------------------------------------------------------
        if configure_s2p:
            config_filename = os.path.join(s2p_output_image_set_dir, '{}_{}_s2p_config.json'.format(ref_image_name_no_extension, sec_image_name_no_extension))
            s2p_output_dir = os.path.join('.', '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

            # create the command to create the s2p config
            create_s2p_configuration_command  = 'python create_s2p_configuration.py'
            create_s2p_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
            create_s2p_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
            create_s2p_configuration_command += ' --gt_image_filename "{}" --out_dir "{}"'
            create_s2p_configuration_command += ' --dsm_resolution {}'
            create_s2p_configuration_command += ' --dsm_radius {}'
            create_s2p_configuration_command += ' --tile_size {}'


            create_s2p_configuration_command = create_s2p_configuration_command.format(s2_config_template_filename,
                                                                                       config_filename,
                                                                                       ref_filename, sec_filename,
                                                                                       gt_filename, s2p_output_dir,
                                                                                       dsm_resolution, dsm_radius, tile_size_s2p)
            # create the configuration
            subprocess.call(create_s2p_configuration_command, shell=True)

            # script to run s2p
            s2p_running_script_list.append('{} {}'.format(s2p_main, config_filename))

            # where to find the dsm
            s2p_dsm_result_list.append('{}'.format( os.path.join(s2p_output_dir, 'dsm.tif')) )



        # --------------------------------------------------------------------------
        # -----S2P-GANET------------------------------------------------------------
        # --------------------------------------------------------------------------
        if configure_s2p_ganet:
            config_filename = os.path.join(s2p_ganet_output_image_set_dir, '{}_{}_s2p_config.json'.format(ref_image_name_no_extension,
                                                                                                    sec_image_name_no_extension))
            s2p_ganet_output_dir = os.path.join('.',
                                          '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

            # create the command to create the s2p config
            create_s2p_configuration_command = 'python create_s2p_configuration.py'
            create_s2p_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
            create_s2p_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
            create_s2p_configuration_command += ' --gt_image_filename "{}" --out_dir "{}"'
            create_s2p_configuration_command += ' --dsm_resolution {}'
            create_s2p_configuration_command += ' --dsm_radius {}'
            create_s2p_configuration_command += ' --tile_size {}'
            create_s2p_configuration_command += ' --disp_range_method {}'

            create_s2p_configuration_command = create_s2p_configuration_command.format(s2_config_template_filename,
                                                                                       config_filename,
                                                                                       ref_filename, sec_filename,
                                                                                       gt_filename, s2p_ganet_output_dir,
                                                                                       dsm_resolution, dsm_radius, tile_size_s2p_ganet,
                                                                                       disp_range_method_s2p_ganet)
            # create the configuration
            subprocess.call(create_s2p_configuration_command, shell=True)

            # script to run s2p
            s2p_ganet_running_script_list.append('{} {}'.format(s2p_ganet_main, config_filename))

            # where to find the dsm
            s2p_ganet_dsm_result_list.append('{}'.format(os.path.join(s2p_ganet_output_dir, 'dsm.tif')))



        #--------------------------------------------------------------------------
        # -----VISSAT COLMAP-------------------------------------------------------
        # --------------------------------------------------------------------------
        if configure_vissat_colmap:
            config_filename = os.path.join(vissat_colmap_output_image_set_dir, '{}_{}_vissat_colmap_config.json'.format(ref_image_name_no_extension,
                                                                                                sec_image_name_no_extension))
            work_dir = os.path.join(vissat_colmap_output_image_set_dir,
                                          '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

            # create the command to create the s2p config
            create_vissat_colmap_configuration_command = 'python create_colmap_configuration.py'
            create_vissat_colmap_configuration_command += ' --config_template_filename "{}" --config_filename "{}"'
            create_vissat_colmap_configuration_command += ' --ref_image_filename "{}" --sec_image_filename "{}"'
            create_vissat_colmap_configuration_command += ' --gt_image_filename "{}" --work_dir "{}"' # --overwrite_work_dir --overwrite_config'

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


        if configure_vissat_planesweep:

            colmap_work_dir = os.path.join(vissat_colmap_output_image_set_dir,
                                    '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

            planesweep_work_dir = os.path.join(vissat_planesweep_output_image_set_dir,
                                    '{}_{}'.format(ref_image_name_no_extension, sec_image_name_no_extension))

            # create the command to create the planesweep
            create_planesweep_colmap_configuration_command = 'python create_planesweep_configuration.py'
            create_planesweep_colmap_configuration_command += ' --planesweep_exe "{}" --plyflatten_exe "{}"'
            create_planesweep_colmap_configuration_command += ' --colmap_work_dir "{}" --planesweep_work_dir "{}"'
            create_planesweep_colmap_configuration_command += ' --gt_image_filename "{}" --ref_img_id {}'
            create_planesweep_colmap_configuration_command += ' --dsm_resolution {}'
            create_planesweep_colmap_configuration_command += ' --dsm_radius {}'

            create_planesweep_colmap_configuration_command = create_planesweep_colmap_configuration_command.format(
                vissat_planesweep_exe,
                plyflatten_exe,
                colmap_work_dir, planesweep_work_dir,
                gt_filename, 0,
                dsm_resolution,
                dsm_radius)

            # create the configuration
            subprocess.call(create_planesweep_colmap_configuration_command, shell=True)

            # script to run vissat colmap
            vissat_planesweep_running_script_list.append('source {}'.format(os.path.join(planesweep_work_dir, 'script.txt')))

            # where to find the dsm
            # <work_dir>/colmap/mvs/dsm/dsm_tif/0000_<ref_image_name>.tif
            vissat_planesweep_dsm_result_list.append(
                '{}'.format(os.path.join(planesweep_work_dir, 'dsm.tif')))

    # save the running scripts, results list and angles list
    if configure_s2p:
        with open(os.path.join(s2p_output_image_set_dir, 'running_script.sh'), 'w') as f:
            for item in s2p_running_script_list:
                f.write("%s\n" % item)
        with open(os.path.join(s2p_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
            for item in s2p_dsm_result_list:
                f.write("%s\n" % item)
        with open(os.path.join(s2p_output_image_set_dir, 'angles.txt'), 'w') as f:
            for item in pair_angles_list:
                f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))

    if configure_s2p_ganet:
        with open(os.path.join(s2p_ganet_output_image_set_dir, 'running_script.sh'), 'w') as f:
            for item in s2p_ganet_running_script_list:
                f.write("%s\n" % item)
        with open(os.path.join(s2p_ganet_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
            for item in s2p_ganet_dsm_result_list:
                f.write("%s\n" % item)
        with open(os.path.join(s2p_ganet_output_image_set_dir, 'angles.txt'), 'w') as f:
            for item in pair_angles_list:
                f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))


    if configure_vissat_colmap:
        with open(os.path.join(vissat_colmap_output_image_set_dir, 'running_script.sh'), 'w') as f:
            for item in vissat_colmap_running_script_list:
                f.write("%s\n" % item)
        with open(os.path.join(vissat_colmap_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
            for item in vissat_colmap_dsm_result_list:
                f.write("%s\n" % item)
        with open(os.path.join(vissat_colmap_output_image_set_dir, 'angles.txt'), 'w') as f:
            for item in pair_angles_list:
                f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))

    if configure_vissat_planesweep:
        with open(os.path.join(vissat_planesweep_output_image_set_dir, 'running_script.sh'), 'w') as f:
            for item in vissat_planesweep_running_script_list:
                f.write("%s\n" % item)
        with open(os.path.join(vissat_planesweep_output_image_set_dir, 'dsm_result_filenames.txt'), 'w') as f:
            for item in vissat_planesweep_dsm_result_list:
                f.write("%s\n" % item)
        with open(os.path.join(vissat_planesweep_output_image_set_dir, 'angles.txt'), 'w') as f:
            for item in pair_angles_list:
                f.write('%s %s %.3f %.3f %.3f %.3f %.3f\n' % tuple(item))


