import numpy as np
import os
import s2p
import rpcm
import utils


if __name__ == '__main__':
    # from skimage.io import imread, imsave
    import argparse

    parser = argparse.ArgumentParser(description='Generate random crops from geotiff')
    parser.add_argument('geotiff_filename', type=str)
    parser.add_argument('crop_directory', type=str)
    parser.add_argument('--crop_prefix', type=str, default='crop')
    parser.add_argument('--crop_count', type=int, default=1)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--random_seed', type=int, default=0)



    args = parser.parse_args(['/vsicurl/http://138.231.80.166:2334/iarpa-2016/training/ground-truth/Challenge1_Lidar.tif',
                              '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA_ERASABLE/crops',
                              '--crop_prefix', 'MVS_gt_crop',
                              '--crop_count', '5',
                              '--width', '400',
                              '--height', '300'])
    np.random.seed(args.random_seed)

    meta, utm_bbox = utils.readGTIFFmeta(args.geotiff_filename)
    gt_width = meta['width']
    gt_height = meta['height']
    crop_width = args.width
    crop_height = args.height
    if crop_width>gt_width:
        raise ValueError('Error: crop_width>gt_width')
    if crop_height>gt_height:
        raise ValueError('Error: crop_height>gt_height')

    # sample x and y of top left vertex
    x_values = np.random.choice(np.arange(gt_width-crop_width), args.crop_count, replace=True )
    y_values = np.random.choice(np.arange(gt_height - crop_height), args.crop_count, replace=True)

    for i in range(args.crop_count):
        x = x_values[i]
        y = y_values[i]
        crop_image_filename = os.path.join(args.crop_directory, '{}_seed_{}_xywh_{}_{}_{}_{}.tif'.format(args.crop_prefix,args.random_seed,x,y,crop_width,crop_height))
        s2p.common.image_crop_gdal(args.geotiff_filename, x, y, crop_width,crop_height, crop_image_filename)