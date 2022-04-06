import argparse
import json
import os
import numpy as np

from utils import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a Vissat Colmap configuration')

    parser.add_argument('--ref_image_filename', required=False, help='Reference image filename')
    parser.add_argument('--sec_image_filename', required=False, help='Secondary image filename')

    parser.add_argument('--gt_image_filename', help='Ground truth image filename')

    parser.add_argument('--image_list_filename', required=False, help='Image list filename')

    parser.add_argument('--config_template_filename', help='Config json template filename (input)')
    parser.add_argument('--config_filename', help='Config json filename (output)')

    parser.add_argument('--work_dir', help='Working directory where to put aoi, images and metas')

    parser.add_argument('--overwrite_config', action='store_true', help='Overwrite existent config')
    parser.add_argument('--overwrite_work_dir', action='store_true', help='Overwrite an existent work_dir')

    parser.add_argument('--aoi_relative_padding', type=float, default=0.1, help='Relative padding for the gt aoi')

    parser.add_argument('--crop_images', action='store_true', help='Crop images to gt aoi')



    # s2p options

    # parser.add_argument('--dsm_resolution', type=float, default=0.5, help='DSM resolution')
    # parser.add_argument('--tile_size', type=int, default=500, help='Tile approximate size')
    # parser.add_argument('--clean_intermediate', action='store_true', help='Clean intermediate files')

    # parse arguments and check
    args = parser.parse_args()



    if os.path.exists(args.config_filename) and not args.overwrite_config:
        raise ValueError('Config filename exists!,  config_filename:"{}"'.format(args.config_filename))

    if os.path.isfile(args.work_dir)  :
        raise ValueError('Out directory name is currently a file!,  work_dir:"{}"'.format(args.work_dir))

    if os.path.isdir(args.work_dir) and not args.overwrite_work_dir:
        raise ValueError('Workdir directory exists!,  work_dir:"{}"'.format(args.work_dir))


    # load the config template
    with open(args.config_template_filename, 'r') as fp:
        config_dict = json.load(fp)

    # load the image list or the ref-sec pair if the list is not set
    if args.image_list_filename is not None:
        with open(args.image_list_filename, 'r') as f:
            image_filenames = f.read().splitlines()
    else:
        if args.ref_image_filename is None or args.sec_image_filename is None:
            raise ValueError('ref-sec pair or list must be set')
        image_filenames = [args.ref_image_filename, args.sec_image_filename]

    angles = get_angles(image_filenames[0], image_filenames[1])
    print(angles)

    # set the values
    aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx = aoi_info_from_geotiff_gt(
        ref_filename=image_filenames[0],
        gt_filename=args.gt_image_filename,
        padding=args.aoi_relative_padding)

    # OBS: dataset_dir holds in COLMAP the NTF and tar files from for example the MVS3D as stored in AWS.
    # In this example we start from geottif images and not from the NTFs. We will skip the two first steps of
    # VisSatSatelliteStereo that is "clean_data" and "crop_image" and we will put the images and the rpcs
    # directly in the folders images and metas of the workdir.
    config_dict['dataset_dir'] = 'N/A'
    config_dict['work_dir'] = args.work_dir

    config_dict['bounding_box']['zone_number'] = zone_number
    config_dict['bounding_box']['hemisphere'] = zone_hemisphere
    config_dict['bounding_box']['ul_easting'] = utm_bbx[0]  # upper left easting == min_easting
    config_dict['bounding_box']['ul_northing'] = utm_bbx[3]  # upper left northing == max_northing
    config_dict['bounding_box']['width'] = utm_bbx[1] - utm_bbx[0]
    config_dict['bounding_box']['height'] = utm_bbx[3] - utm_bbx[2]

    config_dict['alt_min'] = min_height
    config_dict['alt_max'] = max_height

    config_dict['angles'] = angles

    # save the config
    with open(args.config_filename, 'w') as fp:
        json.dump(config_dict, fp, indent=2)


    #save aoi, images and metas to the work_dir
    if not os.path.isdir(args.work_dir) or args.overwrite_work_dir:
        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(os.path.join(args.work_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.work_dir, 'metas'), exist_ok=True)

    aoi_dict = {}
    aoi_dict['ul_easting'] = utm_bbx[0]
    aoi_dict['lr_easting'] = utm_bbx[1]
    aoi_dict['lr_northing'] = utm_bbx[2]
    aoi_dict['ul_northing'] = utm_bbx[3]
    aoi_dict['width'] = utm_bbx[1] - utm_bbx[0]
    aoi_dict['height'] = utm_bbx[3] - utm_bbx[2]
    aoi_dict['zone_number'] = zone_number
    aoi_dict['hemisphere'] = zone_hemisphere
    aoi_dict['lon_min'] = lonlat_bbx[0]
    aoi_dict['lon_max'] = lonlat_bbx[1]
    aoi_dict['lat_max'] = lonlat_bbx[3]
    aoi_dict['lat_min'] = lonlat_bbx[2]
    aoi_dict['alt_min'] = min_height
    aoi_dict['alt_max'] = max_height

    # save the aoi in vissat colmap format
    with open(os.path.join(args.work_dir, 'aoi.json'), 'w') as fp:
        json.dump(aoi_dict, fp, indent=2)

    # save images and metas
    for i, image_filename in enumerate(image_filenames):
        _, image_name = os.path.split(image_filename)
        image_name_no_extension, extension = os.path.splitext(image_name)

        image_name_no_extension_numbered = '{:04d}_{}'.format(i, image_name_no_extension)

        # filename to save the normalized image
        image_filename_png = os.path.join(args.work_dir, 'images', image_name_no_extension_numbered + '.png')
        # filename to save the meta in json format
        image_filename_meta_json = os.path.join(args.work_dir, 'metas', image_name_no_extension_numbered + '.json')

        if args.crop_images:
            # CROP
            crop_tmp_filename = 'TMP_CROP_KK.tif'
            rpc = rpcm.rpc_from_geotiff(image_filename)
            x, y, w, h = rpcm.utils.bounding_box_of_projected_aoi(rpc, aoi, z=min_height)
            s2p.common.image_crop_gdal(image_filename, x, y, w, h, crop_tmp_filename)
            image_filename = crop_tmp_filename

        # load and normalize image
        I = s2p.common.gdal_read_as_array_with_nans(image_filename)
        J = normalize_image(I)

        # load the rpc info from the image
        vissat_meta_dict = vissat_meta_from_geotiff(image_filename)

        # save the normalized image
        s2p.common.rasterio_write(image_filename_png, J)

        # save the rpc
        with open(image_filename_meta_json, 'w') as fp:
            json.dump(vissat_meta_dict, fp, indent=2)