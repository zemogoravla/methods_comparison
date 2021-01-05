import argparse
import glob
import json
import os
import numpy as np

from utils import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a Vissat Planesweep configuration from a Vissat Colmap workdir result')
    parser.add_argument('--planesweep_exe', help='Vissat Planesweep executable')
    parser.add_argument('--plyflatten_exe', help='Plyflatten executable')
    parser.add_argument('--colmap_work_dir', help='Vissat colmap working dir')
    parser.add_argument('--planesweep_work_dir', help='Planesweep working dir')
    parser.add_argument('--gt_image_filename', help='Ground truth image filename')

    parser.add_argument('--ref_img_id', help='Reference image id')
    parser.add_argument('--dsm_resolution', type=float, default=0.5, help='DSM resolution')
    parser.add_argument('--dsm_radius', type=int, default=0, help='DSM radius')

    parser.add_argument('--overwrite_work_dir', action='store_true', help='Overwrite an existent work_dir')

    # parse arguments and check
    args = parser.parse_args()


    if os.path.isfile(args.planesweep_work_dir)  :
        raise ValueError('Out directory name is currently a file!,  planesweep_work_dir:"{}"'.format(args.planesweep_work_dir))

    if os.path.isdir(args.planesweep_work_dir) and not args.overwrite_work_dir:
        raise ValueError('Workdir directory exists!,  planesweep_work_dir:"{}"'.format(args.planesweep_work_dir))

    colmap_workdir = args.colmap_work_dir
    planesweep_workdir = args.planesweep_work_dir

    image_folder = os.path.join(colmap_workdir, 'colmap/mvs/images')
    cameras_dict_json = os.path.join(colmap_workdir, 'colmap/sfm_perspective/', 'init_ba_camera_dict.json')
    image_list_csv = os.path.join(planesweep_workdir, 'image_info_list.csv')

    # load the cameras of colmap sfm
    with open(cameras_dict_json, 'r') as fp:
        cameras_dict = json.load(fp)

    # initialize the image info list
    # will hold the [img_id, img_name, fx, fy, cx, cy, s, qw, qx, qy, qz, tx, ty, tz] per image
    image_info_list = []

    # image filenames in the colmap/mvs/images folder
    image_filenames = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    for i, image_filename in enumerate(image_filenames):
        _, image_name = os.path.split(image_filename)
        image_id = int(image_name.split('_')[0])

        camera_info = cameras_dict[image_name]

        # image_info:  [img_id, img_name, fx, fy, cx, cy, s, qw, qx, qy, qz, tx, ty, tz]
        image_info = [image_id, image_name, *camera_info[2:]]

        # append to the image_info_list
        image_info_list.append(image_info)

    # save the information for all the images in the image_info_list.csv file
    if os.path.isdir(planesweep_workdir):
        raise ValueError('PlaneSweep workdir already exist !!')

    if not os.path.isdir(planesweep_workdir):
        os.makedirs(planesweep_workdir)

    np.savetxt(image_list_csv, image_info_list, fmt='%s')

    # #---------------------------------------------------------------------------
    # # load the colmap aoi
    # colmap_aoi_filename = os.path.join(args.colmap_work_dir, 'aoi.json')
    # with open(colmap_aoi_filename, 'r') as fp:
    #     aoi_dict = json.load(fp)
    #
    # #---------------------------------------------------------------------------

    # save the configuration script

    output_folder = os.path.join(planesweep_workdir, 'output')
    script_filename = os.path.join(planesweep_workdir, 'script.txt')

    min_height, max_height = get_min_max_height_from_gt(args.gt_image_filename)
    # guard
    min_height += -5
    max_height += 5
    # Vissat planesweep has an offset of +30   ?????????
    min_depth = min_height -70  #+ 30
    max_depth = max_height +70  #+ 30


    command = args.planesweep_exe
    command += ' --imageFolder ' + image_folder
    command += ' --imageList ' + image_list_csv
    command += ' --outputFolder ' + output_folder
    command += ' --refImgId ' + str(args.ref_img_id)
    command += ' --matchingCost census --windowRadius 2'
    command += ' --nX 0 --nY 0 --nZ 1 '
    command += ' --firstD {} --lastD {} --numPlanes {} '.format(min_depth, max_depth,
                                                                int(np.ceil((max_depth - min_depth) / 0.25)))
    command += ' --filterCostVolume 1 --guidedFilterRadius 5 --guidedFilterEps 100 '
    command += ' --saveCostVolume 0 --debug 0 --saveBest 1 '
    command += ' --filter 0 --filterThres 1 '
    command += ' --saveXYZMap 1 --savePointCloud 1'

    command += '\n'
    command += 'ls {} | {} -radius {} {} {}'.format(os.path.join(output_folder, 'best', 'best_point_cloud.ply'),
                                         args.plyflatten_exe,
                                         args.dsm_radius,
                                         args.dsm_resolution,
                                         os.path.join(planesweep_workdir, 'dsm.tif'))

    with open(script_filename, 'w') as fp:
        fp.write(command)

