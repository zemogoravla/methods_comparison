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
from modify_rectification import modify_rectification


def point_is_in_polygon(point, polygon):
    number_of_vertices = polygon.shape[0]

    x = point[0]
    y = point[1]
    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])
    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])

    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False

    # https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    inside = False
    for i in range(number_of_vertices):
        if i == 0:
            j = number_of_vertices - 1
        else:
            j = i - 1

        if ((polygon[i, 1] > y) != (polygon[j, 1] > y)) and x < (polygon[j, 0] - polygon[i, 0]) * (
                y - polygon[i, 1]) / (polygon[j, 1] - polygon[i, 1]) + polygon[i, 0]:
            inside = not inside

    return inside


def polygon_is_in_polygon(test_polygon, polygon):  # convex polygons
    for i in range(test_polygon.shape[0]):
        if not point_is_in_polygon(test_polygon[i, :], polygon):
            return False
    return True


def get_crop_origin(img, width, height, debug=False):
    Point = namedtuple('Point', 'x y')  # x=col, y=row
    # find starting rows and cols by projecting
    row_projection = ~ np.isnan(np.nanmax(img, axis=0))  # boolean False is all the column is nan
    col_projection = ~ np.isnan(np.nanmax(img, axis=1))  # boolean False is all the row is nan
    # limits of the rotated rectangle
    first_row = np.where(col_projection)[0][0]
    last_row = np.where(col_projection)[0][-1]
    first_col = np.where(row_projection)[0][0]
    last_col = np.where(row_projection)[0][-1]

    # find the vertices corresponding to first col and first row
    v0 = Point(first_col, np.where(~ np.isnan(img[:, first_col]))[0][0])
    v1 = Point(np.where(~ np.isnan(img[first_row, :]))[0][0], first_row)
    v2 = Point(last_col, np.where(~ np.isnan(img[:, last_col]))[0][0])
    v3 = Point(np.where(~ np.isnan(img[last_row, :]))[0][0], last_row)

    polygon = np.array([v0, v1, v2, v3])

    found = False
    for left in range(first_col, last_col-width+1):
        if found:
            break
        for top in range(first_row, last_row-height+1):
            test_polygon = np.array(
                [[left, top], [left + width, top], [left + width - 1, top + height - 1], [left, top + height - 1]])
            if polygon_is_in_polygon(test_polygon, polygon):
                found = True
                break

    if debug:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(polygon[:, 0], polygon[:, 1], 'r*-')

    if found:
        if debug:
            print('Polygon:', test_polygon)
            plt.plot(test_polygon[:, 0], test_polygon[:, 1], 'g*-')
            plt.show()
        return test_polygon[0]
    else:
        if debug:
            print('NO Polygon')
        return -1, -1


def get_crops(left_chip_filename, right_chip_filename, disp_chip_filename, mask_chip_filename, crop_width, crop_height,
              disp_guard, debug=False):
    '''
    From an original chip of the RVL dataset, it crops a chip that is inside the valid region
    It takes into account that the disp_chip has NaN outside the valid region
    :param left_chip_filename:
    :param right_chip_filename:
    :param disp_chip_filename:
    :param mask_chip_filename:
    :param crop_width:
    :param crop_height:
    :param disp_guard: Min disparity we will have in the output chips
    :param debug:
    :return: Returns the left, right, disp and mask crops. It also returns the left ant top of the crop and
    the left and right translations imposed to the chips with respect to the originals
    Left translation affects left, right and disp chips. Right translation affects the right chip. Whole tranlation
    affects the disparity values.
    '''
    L = imread(left_chip_filename)
    R = imread(right_chip_filename)
    D = imread(disp_chip_filename)
    M = imread(mask_chip_filename)


    # # ESTO ERA NECESARIO ANTES DE REALIZAR LA RE-RECTIFICACION-------------------------
    # min_disp = np.nanmin(D)
    # max_disp = np.nanmax(D)
    # translation = int(np.ceil((max(max_disp, 0) + disp_guard) / 2) * 2)
    # L_translation = 0  # direct mapping !   (DO NOT TOUCH: using 0 to simplify "get_crop_origin")
    # R_translation = -translation  # direct mapping !   (the resulting chip will move to the left w.r.t the original)
    #
    # # tform_left_image = EuclideanTransform(translation=(0, 0) ) # inverse mapping  (No movemos la imagen izquierda/disp/mask)
    # tform_right_image = EuclideanTransform(translation=(-R_translation, 0))  # inverse mapping !!!
    # LL = L  # LL = warp(L, tform_left_image)
    # RR = warp(R, tform_right_image)
    # DD = D  # DD = warp(D, tform_left_image)
    # MM = M  # MM = warp(M, tform_left_image)
    #
    # # values of D change with the translation
    # DD = DD - translation
    # # ESTO ERA NECESARIO ANTES DE REALIZAR LA RE-RECTIFICACION-------------------------

    LL = L
    RR = R
    DD = D
    MM = M
    L_translation, R_translation = 0 , 0

    # fit the crop in DD, returns None if the crop does not fit
    left, top = get_crop_origin(DD, width=crop_width, height=crop_height)
    if left < 0:
        # No se pudo sacar el crop
        return None, None, None, None, None, None, None, None

    DD_crop = DD[top:top + crop_height, left:left + crop_width]
    LL_crop = LL[top:top + crop_height, left:left + crop_width]
    RR_crop = RR[top:top + crop_height, left:left + crop_width]
    MM_crop = MM[top:top + crop_height, left:left + crop_width]

    if debug:
        plt.figure()
        plt.imshow(DD_crop)
        plt.figure()
        plt.imshow(LL_crop)
        plt.title('LL_crop')
        plt.figure()
        plt.imshow(RR_crop)
        plt.title('RR_crop')
        plt.figure()
        plt.imshow(MM_crop)
        plt.title('MM_crop')
        plt.show(block=True)

    return LL_crop, RR_crop, DD_crop, MM_crop, left, top, L_translation, R_translation


def get_rl_chip_filename(lr_chip_filename):
    lr_name = os.path.basename(lr_chip_filename)[:-len('_Rectified.tif')]
    # print(lr_name)
    [left_name, right_name] = lr_name.split('_and_')
    rl_name = right_name + '_and_' + left_name
    # print(rl_name)
    rl_filename = os.path.join(chips_directory, rl_name + '_Rectified.tif')
    return rl_filename


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Crop chips and disparity to build de database to train ganet')
    parser.add_argument('rvl_base_directory', type=str)
    parser.add_argument('rvl_left_right_base_directory', type=str)
    parser.add_argument('rvl_dataset_name', type=str, help='One of: Explorer, JAX, MP1, MP2, MP3, PS1, PS2, PS3')
    parser.add_argument('--crop_width', type=int, default=768)
    parser.add_argument('--crop_height', type=int, default=384)
    parser.add_argument('--disp_guard', type=int, default=3)
    parser.add_argument('--tmp_dir', type=str, default='./TMP_DIR_MAKE_CROPS_FROM_RVL_DATASET')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    original_dataset_directory = os.path.join(args.rvl_base_directory, args.rvl_dataset_name)
    left_right_dataset_directory = os.path.join(args.rvl_left_right_base_directory, args.rvl_dataset_name)


    if os.path.exists(left_right_dataset_directory) and not args.overwrite:
        raise ValueError('Directotio de salida ya existe: {}'.format(left_right_dataset_directory))

    # temporary directory and files
    os.makedirs(args.tmp_dir, exist_ok=True)
    tmp_left_filename = os.path.join(args.tmp_dir, 'left.tif')
    tmp_right_filename = os.path.join(args.tmp_dir, 'right.tif')
    tmp_disp_filename = os.path.join(args.tmp_dir, 'disp.tif')
    tmp_mask_filename = os.path.join(args.tmp_dir, 'mask.tif')


    chips_directory = os.path.join(original_dataset_directory, 'Rectified_Chips')
    disparity_directory = os.path.join(original_dataset_directory, 'Disparity')
    masks_directory = os.path.join(original_dataset_directory, 'Masks')
    metadata_directory = os.path.join(original_dataset_directory, 'Metadata')

    left_output_directory = os.path.join(left_right_dataset_directory, 'Left')
    right_output_directory = os.path.join(left_right_dataset_directory, 'Right')
    mask_output_directory = os.path.join(left_right_dataset_directory, 'Masks')
    disparity_output_directory = os.path.join(left_right_dataset_directory, 'Disparity')
    metadata_output_directory = os.path.join(left_right_dataset_directory, 'Metadata')

    os.makedirs(left_output_directory, exist_ok=True)
    os.makedirs(right_output_directory, exist_ok=True)
    os.makedirs(mask_output_directory, exist_ok=True)
    os.makedirs(disparity_output_directory, exist_ok=True)
    os.makedirs(metadata_output_directory, exist_ok=True)

    chips_all_filenames = sorted(glob.glob(os.path.join(chips_directory, '*_Rectified.tif')))

    for lr_chip_filename in chips_all_filenames:
        rl_chip_filename = get_rl_chip_filename(lr_chip_filename)
        disparity_filename = lr_chip_filename.replace('Rectified_Chips', 'Disparity').replace('_Rectified.tif',
                                                                                              '_disp.tif')
        mask_filename = lr_chip_filename.replace('Rectified_Chips', 'Masks').replace('_Rectified.tif', '_mask.png')
        metadata_filename = lr_chip_filename.replace('Rectified_Chips', 'Metadata').replace('_Rectified.tif',
                                                                                            '_metadata.json')

        # todos con el mismo nombre, cambian los directorios
        left_output_filename = lr_chip_filename.replace(chips_directory, left_output_directory).replace(
            '_Rectified.tif', '.tif')
        right_output_filename = left_output_filename.replace('Left', 'Right')
        disp_output_filename = left_output_filename.replace('Left', 'Disparity')
        mask_output_filename = left_output_filename.replace('Left', 'Masks').replace('.tif', '.png')
        metadata_output_filename = left_output_filename.replace('Left', 'Metadata').replace('.tif', '.json')


        # modify rectification first
        modify_rectification(lr_chip_filename, rl_chip_filename, disparity_filename, mask_filename,
                             tmp_left_filename, tmp_right_filename, tmp_disp_filename, tmp_mask_filename,
                             register_horizontally_shear=True,
                             register_horizontally_translation=True,
                             register_horizontally_translation_flag='negative',
                             debug=False
                             )

        # get crops from modified images
        L_crop, R_crop, D_crop, M_crop, crop_left, crop_top, L_translation, R_translation = \
            get_crops(tmp_left_filename, tmp_right_filename, tmp_disp_filename, tmp_mask_filename,
                      crop_width=args.crop_width, crop_height=args.crop_height, disp_guard=args.disp_guard)

        if not L_crop is None:
            imsave(left_output_filename, L_crop.astype(np.float32))
            imsave(right_output_filename, R_crop.astype(np.float32))
            imsave(disp_output_filename, -D_crop.astype(np.float32)) # HAY QUE GUARDAR CON VALORES POSITIVOS POR ESO EL SIGNO !!!!!!
            imsave(mask_output_filename, M_crop.astype(np.uint8))
            shutil.copy2(metadata_filename, metadata_output_filename)
            print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(crop_left, crop_top, args.crop_width,
                                                                           args.crop_height, L_translation,
                                                                           R_translation,
                                                                           lr_chip_filename, rl_chip_filename,
                                                                           disparity_filename, mask_filename,
                                                                           metadata_filename,
                                                                           left_output_filename, right_output_filename,
                                                                           disp_output_filename, mask_output_filename,
                                                                           metadata_output_filename))
        else:
            print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(-1, -1, args.crop_width,
                                                                           args.crop_height, -1, -1,
                                                                           lr_chip_filename, rl_chip_filename,
                                                                           disparity_filename, mask_filename,
                                                                           metadata_filename,
                                                                           left_output_filename, right_output_filename,
                                                                           disp_output_filename, mask_output_filename,
                                                                           metadata_output_filename))