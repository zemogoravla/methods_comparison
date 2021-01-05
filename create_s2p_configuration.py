import argparse
import json
import os

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create an S2P configuration file for an image pair + gt')

    parser.add_argument('--ref_image_filename',  help='Reference image filename')
    parser.add_argument('--sec_image_filename',  help='Secondary image filename')
    parser.add_argument('--gt_image_filename', help='Ground truth image filename')

    parser.add_argument('--config_template_filename', help='Config json template filename (input)')
    parser.add_argument('--config_filename', help='Config json filename (output)')

    parser.add_argument('--out_dir', help='Output directory for the s2p run')

    parser.add_argument('--overwrite_config', action='store_true', help='Overwrite existent config')
    parser.add_argument('--allow_existent_out_dir', action='store_true', help='Allow in the config an existent out_dir')

    parser.add_argument('--aoi_relative_padding', type=float, default=0.1, help='Relative padding for the gt aoi')

    # s2p options

    parser.add_argument('--dsm_resolution', type=float, default=0.5, help='DSM resolution')
    parser.add_argument('--tile_size', type=int, default=500, help='Tile approximate size')
    parser.add_argument('--clean_intermediate', action='store_true', help='Clean intermediate files')
    parser.add_argument('--dsm_radius', type=int, default=0, help='DSM radius')


    # parse arguments and check
    args = parser.parse_args()

    angles = get_angles(args.ref_image_filename, args.sec_image_filename)
    print(angles)

    if os.path.exists(args.config_filename) and not args.overwrite_config:
        raise ValueError('Config filename exists!,  config_filename:"{}"'.format(args.config_filename))

    if os.path.isfile(args.out_dir)  :
        raise ValueError('Out directory name is currently a file!,  out_dir:"{}"'.format(args.out_dir))

    if os.path.isdir(args.out_dir) and not args.allow_existent_out_dir:
        raise ValueError('Out directory exists!,  out_dir:"{}"'.format(args.out_dir))


    # load the config template
    with open(args.config_template_filename, 'r') as fp:
        config_dict = json.load(fp)

    # set the values
    aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
        ref_filename=args.ref_image_filename,
        gt_filename=args.gt_image_filename,
        padding=args.aoi_relative_padding)

    config_dict['out_dir'] = args.out_dir
    config_dict['images'][0]['img'] = args.ref_image_filename
    config_dict['images'][1]['img'] = args.sec_image_filename

    config_dict['dsm_resolution'] = args.dsm_resolution
    config_dict['tile_size'] = args.tile_size
    config_dict['clean_intermediate'] = args.clean_intermediate
    config_dict['dsm_radius'] = args.dsm_radius

    config_dict['utm_bbx'] = utm_bbx
    config_dict['ll_bbx'] = lonlat_bbx
    config_dict['utm_zone'] = '{}{}'.format(zone_number, zone_hemisphere)

    config_dict['angles'] = angles

    # save the config
    with open(args.config_filename, 'w') as fp:
        json.dump(config_dict, fp, indent=2)
