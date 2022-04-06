import os
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sli
import utm
import srtm4
import skimage
from skimage.io import imread,imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.registration import phase_cross_correlation
from scipy import ndimage as ndi

from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from numba import jit
#import pandas as pd

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8


import glob
import shutil
import time
import datetime

from skimage.transform import warp,warp_coords,EuclideanTransform, SimilarityTransform, AffineTransform
from skimage import transform as tf

import time

#--------------------------------------------------------------------------------------------


#sys.path.append("../imsat_tools/libTP/")

# Modules from the TPs
#import utils
#import stereo
#import rectification
#import vistools
#import rpc_model
#import triangulation
#--------------------------------------------------------------------------------------------

sys.path.append("/home/agomez/ownCloud/Documents/doctorado/MultiViewStereo/python/imsat_tools/")
from tools.registration import register_images_with_nccshift, apply_nccshift_transformation, interpolate_image_with_bdint
from tools.benchmark import compute_benchmark
from tools.tiling import tile_image, untile_image
import tools.diffusion
from tools.disparity import compute_mgm_multi
from tools.dataset import *
#--------------------------------------------------------------------------------------------

def register_images_with_phase_cross_correlation(ref, mov):
    # adjust sizes for the original and operative images
    rows = max(ref.shape[0], mov.shape[0])
    cols = max(ref.shape[1], mov.shape[1])
    # create the big images initialized with NaN
    ref_resized = np.ones((rows, cols)) * np.nan
    mov_resized = np.ones((rows, cols)) * np.nan
    # reference_image_resized = np.ones((rows, cols)) * np.nan
    # moving_image_resized = np.ones((rows, cols)) * np.nan

    ref_resized[:ref.shape[0], :ref.shape[1]] = ref
    mov_resized[:mov.shape[0], :mov.shape[1]] = mov

    ref_mask = ~ np.isnan(ref_resized)
    mov_mask = ~ np.isnan(mov_resized)

    detected_shift = phase_cross_correlation(ref_resized, mov_resized,
                                             reference_mask=ref_mask,
                                             moving_mask=mov_mask,
                                             )

    # detected_shift is (rows, cols) and is the translation to apply to the mov image.
    # returned transformation is compatible with nccshift [[dx,dy,scale,dz]] , dx = -detected_shift[1], dy = -detected_shift[0]
    registration_transform = np.array([[0.0, 0.0, 1.0, 0.0]])
    registration_transform[0][0] = -detected_shift[1]
    registration_transform[0][1] = -detected_shift[0]

    #[ [ -detected_shift[1], -detected_shift[0], 1., 0.] ]
    #apply shift
    registered_image_resized = apply_nccshift_transformation(mov_resized, registration_transform[0])

    # register Z
    dz = np.nanmedian(ref_resized - registered_image_resized)
    registration_transform[0][3] += dz
    registered_image_resized = registered_image_resized + dz

    return registered_image_resized, registration_transform


def register_images_in_height(ref, mov, initial_transformation):
    # adjust sizes for the original and operative images
    rows = max(ref.shape[0], mov.shape[0])
    cols = max(ref.shape[1], mov.shape[1])
    # create the big images initialized with NaN
    ref_resized = np.ones((rows, cols)) * np.nan
    mov_resized = np.ones((rows, cols)) * np.nan
    # reference_image_resized = np.ones((rows, cols)) * np.nan
    # moving_image_resized = np.ones((rows, cols)) * np.nan

    ref_resized[:ref.shape[0], :ref.shape[1]] = ref
    mov_resized[:mov.shape[0], :mov.shape[1]] = mov

    registration_transform = initial_transformation
    registered_image_resized = apply_nccshift_transformation(mov_resized, registration_transform[0])

    # register Z
    dz = np.nanmedian(ref_resized - registered_image_resized)
    registration_transform[0][3] += dz
    registered_image_resized = registered_image_resized + dz

    return registered_image_resized, registration_transform



#def pandas_dataframe_results(results, column_names):
#    pd.options.display.float_format = '{:,.3f}'.format
#    results_df = pd.DataFrame(results)
#    results_df.columns = column_names
#    #['evaluated', 'bad', 'invalid', 'totalbad', 'avgAbsErr','accuracy', 'completeness', 'dispersion']
#    return results_df
#--------------------------------------------------------------------------------------------
data_base_directory_cluster = '/clusteruy/home/agomez/DATA/'
data_base_directory_local = '/media/agomez/SeagateGoFlex750GB/SATELITE/DATA_CLUSTER/'
data_base_directory = data_base_directory_local

title = 'GANET_WHU_EPOCH_45' #'GANET_RVL_EPOCH_65_BIS' # 'GANET_WHU_EPOCH_45_BIS'  # 'GANET_WHU_EPOCH_32'
prefix = 'MVS'
nit_or_fit = 'NIT'

base_directory = os.path.join( data_base_directory, 'COMPARISON_{}_{}/'.format(prefix, nit_or_fit) )
gt_base_directory = os.path.join( data_base_directory,'{}_DATA/gt'.format(prefix) )
class_base_directory = os.path.join( data_base_directory, '{}_DATA/CLASS'.format(prefix) )
#results_csv_filename = '/clusteruy/home/agomez/DATA/COMPARISON_{}_{}/results_s2p_ganet_20210112.csv'.format(prefix,nit_or_fit)

image_set_numbers_JAX = [156, 165, 214, 251, 264] # JAX
image_set_numbers_OMA = [203, 247, 251, 287, 353] # OMA
image_set_numbers_MVS = [1, 2, 3, 4, 5] # MVS

#image_set_numbers_JAX =[156]

if prefix == 'JAX':
    image_sets = ['{}_{:03d}'.format(prefix, i)  for i in image_set_numbers_JAX]
elif prefix == 'OMA':
    image_sets = ['{}_{:03d}'.format(prefix, i)  for i in image_set_numbers_OMA]
elif prefix == 'MVS':
    image_sets = ['mvs3d_gt_crop_{:02d}'.format(i)  for i in image_set_numbers_MVS]
else:
    raise ValueError('Invalid prefix')

dsm_filenames_list=[]
test_name_list = []
image_names_and_angles_list = []  # ref_name, sec_name, ref_zenith, ref_azimut, sec_zenith, sec_azimut, ref_sec_angle
method_and_image_set_list = []

gt_filenames_list = []
class_filenames_list=[]

for method in  ['s2p_ganet_whu_epoch_45']:  #['s2p_ganet_whu_epoch_45']: #['s2p_ganet_rvl_epoch_65']: #['s2p_ganet_whu_epoch_32']:  #['s2p_ganet_tile_1872_480']:  # ['vissat_colmap']:  ['s2p', 's2p_ganet', 'vissat_colmap']:
    print(method)
    for image_set in image_sets:
        print(image_set)
        dsm_results_filename = os.path.join(base_directory,
                                            '{}'.format('s2p_ganet_rvl_epoch_65'),        #'{}'.format(method),
                                            '{}'.format(image_set),
                                            'dsm_result_filenames.txt')
        angles_filename = os.path.join(base_directory,
                                            '{}'.format('s2p_ganet_rvl_epoch_65'),        #'{}'.format(method),
                                            '{}'.format(image_set),
                                            'angles.txt')
        if prefix in ['JAX', 'OMA']:
            gt_filename = os.path.join(gt_base_directory, '{}_DSM.tif'.format(image_set))
        elif prefix=='MVS':
            gt_filename = os.path.join(gt_base_directory, '{}.tif'.format(image_set))

        class_filename = os.path.join(class_base_directory, '{}_CLS.tif'.format(image_set))
        if os.path.isfile(dsm_results_filename):
            dsm_filenames = list(np.loadtxt(dsm_results_filename, dtype=str))
        else:
            dsm_filenames = sorted(glob.glob(os.path.join(base_directory,method,image_set,'**','dsm.tif')))

        dsm_filenames_modif = [f.replace(data_base_directory_cluster, data_base_directory) for f in dsm_filenames]
        dsm_filenames = dsm_filenames_modif

        if not os.path.isabs(dsm_filenames[0]):
            dsm_results_base_directory = os.path.dirname(dsm_results_filename)
            dsm_filenames = [os.path.join(dsm_results_base_directory, f) for f in dsm_filenames]
        # PANDAS  image_names_and_angles = pd.read_csv(angles_filename,sep=' ',header=None)
        image_names_and_angles = np.genfromtxt(angles_filename,  delimiter=' ', dtype=None)


        dsm_filenames_list += dsm_filenames
        # PANDAS  image_names_and_angles_list += image_names_and_angles.values.tolist()
        image_names_and_angles_list += [list(X) for X in list(image_names_and_angles)]  #from a list of tuples to a list of lists

        method_and_image_set_list += ( [ [method, image_set] for d in dsm_filenames ])
        gt_filenames_list += [gt_filename for d in dsm_filenames ]
        class_filenames_list += [class_filename for d in dsm_filenames ]

dsm_checked_filenames = []
invalid_indices = []
for i,dsm_filename in enumerate(dsm_filenames_list):
    if os.path.isfile(dsm_filename):
        dsm_checked_filenames.append(dsm_filename)
    else:
        dsm_dir, dsm_name = os.path.split(dsm_filename)
        dsm_real_filename = glob.glob(os.path.join(dsm_dir,'*.tif'))
        if len(dsm_real_filename) == 1:
            dsm_checked_filenames.append(dsm_real_filename[0])
        else:
            dsm_checked_filenames.append(None)
            invalid_indices.append(i)


#print(dsm_checked_filenames)
print('PROBLEM -------')
print(invalid_indices)
print([dsm_filenames_list[i] for i in invalid_indices] )
print('---------------')

print(len(dsm_filenames_list))
print(len(image_names_and_angles_list))
print(len(method_and_image_set_list))
print(len(gt_filenames_list))
print(len(class_filenames_list))

dsm_checked_filenames = [dsm_checked_filenames[i] for i in range(len(dsm_filenames_list)) if i not in invalid_indices]
image_names_and_angles_list = [image_names_and_angles_list[i] for i in range(len(dsm_filenames_list)) if i not in invalid_indices]
method_and_image_set_list = [method_and_image_set_list[i] for i in range(len(dsm_filenames_list)) if i not in invalid_indices]
gt_filenames_list = [gt_filenames_list[i] for i in range(len(dsm_filenames_list)) if i not in invalid_indices]
class_filenames_list = [class_filenames_list[i] for i in range(len(dsm_filenames_list)) if i not in invalid_indices]

print(len(dsm_checked_filenames))
print(len(image_names_and_angles_list))
print(len(method_and_image_set_list))
print(len(gt_filenames_list))
print(len(class_filenames_list))

#--------------------------------------------------------------------------------------------

# # PARAMETROS


if prefix == 'JAX':
    fixed_transformation = np.array([[51., 52., 1., 0.]])   #JAX
elif prefix == 'OMA':
    fixed_transformation = np.array([[53., 53., 1., 0.]])   #OMA
elif prefix == 'MVS':
    fixed_transformation = np.array([[53., 53., 1., 0.]])   #
else:
    raise ValueError('Invalid prefix')


registration_algorithm = 'nccshift' # nccshift, pcc, fixed
reuse_registration = False

bdint_interpolate_moving_image = True
bdint_min_max_avg_option='avg'
adjust_dz_method='nanmedian'

registration_pixel_range = 11
z_tolerance=1
keep_z = False
show_figures = False #True False

use_classification_mask=False
# LAS CLASSIFICATION CODES
# 0:Never classified, 1:Unassigned, 2:Ground, 3:Low Vegetation, 4:Medium Vegetation, 5:High Vegetation,
# 6:Building, 7:Low Point, 8:Reserved *, 9:Water, 10:Rail, 11:Road Surface, 12:Reserved *,
# 13:Wire - Guard (Shield), 14:Wire - Conductor (Phase), 15:Transmission Tower, 16:Wire-Structure Connector (Insulator),
# 17:Bridge Deck, 18:High Noise, 19-63:Reserved, 64-255:User Definable
classification_mask_accept_list = [2,6,11,17]            # reject_list y accept_list son excluyentes
classification_mask_accept_list = [2]            # reject_list y accept_list son excluyentes

classification_mask_reject_list = []                     # una lista o la otra


results_column_names = ['method', 'set', 'ref', 'sec',
                        'evaluated', 'bad', 'invalid', 'totalbad', 'completeness', 'accuracy', 'meanAE', 'medianAE',
                        'tx', 'ty','scale','tz',
                        'ref_zenith', 'ref_azimut', 'sec_zenith', 'sec_azimut', 'ref_sec_angle']
results = []


test_image_filename_list = dsm_checked_filenames
test_info_list = [ method_and_image_set_list[i] + image_names_and_angles_list[i][:2] for i in range(len(method_and_image_set_list))]
test_angles_list = [image_names_and_angles_list[i][2:] for i in range(len(method_and_image_set_list))]

# test_name_list = ['dsm']

indices_to_run = range(len(test_image_filename_list))
#indices_to_run=[6,30+6]# solo para debug
#--------------------------------------------------------------------------------------------

np.warnings.filterwarnings('ignore')


for ordinal,i in enumerate(indices_to_run):
    print('Procesando {:03d}/{:03d}'.format(ordinal, len(indices_to_run) ) )
    test_image_filename = test_image_filename_list[i]
    gt_filename = gt_filenames_list[i]

    test_info = test_info_list[i]
    test_angles = test_angles_list[i]

    reference_image = imread(gt_filename)
    test_image = imread(test_image_filename)


    print('---------------------------------------------------------------')
    print(test_info)
    print('---------------------------------------------------------------')


    # benchmark does not tolerate INF
    test_image[np.isinf(test_image)] = np.nan
    # colmap uses -10000 for undefined pixels
    test_image[test_image ==-10000] = np.nan


    # register

    if registration_algorithm=='fixed':
        print('Fixed registration')
        transformation = fixed_transformation.copy()
        aux, transformation = register_images_in_height(reference_image, test_image, transformation)

    elif (reuse_registration and i>0):
        # si metodo y set son los mismos reusar el registrado
        print('Using previous registration')
        test_image = apply_nccshift_transformation(test_image,transformation[0])

    else:
        if registration_algorithm == 'nccshift':
            aux, transformation = register_images_with_nccshift(reference_image,
                                                          test_image,
                                                          pixel_range=registration_pixel_range,
                                                               bdint_interpolate_moving_image=bdint_interpolate_moving_image,
                                                                bdint_min_max_avg_option=bdint_min_max_avg_option,
                                                                 adjust_dz_method=adjust_dz_method)
        elif registration_algorithm == 'pcc':
            aux, transformation = register_images_with_phase_cross_correlation(reference_image, test_image)
        else:
            raise ValueError('Not a valid registration algorithm: {}'.format(registration_algorithm))

        print(transformation)

    if keep_z:
        transformation[0][3]=0
        test_image = apply_nccshift_transformation(test_image,transformation[0])
        print('Keeping z unchanged')
    else:
        test_image = aux


    print('Registration transform:',transformation)



    # compute benchmark

    if use_classification_mask:
        class_filename = class_filenames_list[i]
        class_image = imread(class_filename)
        if classification_mask_accept_list: #if not empty (implicit booleanness)
            mask_image = np.zeros_like(reference_image,dtype=np.uint8)
            for c in classification_mask_accept_list:
                mask_image[class_image==c]=255
        if classification_mask_reject_list: #if not empty (implicit booleanness)
            mask_image = np.ones_like(reference_image,dtype=np.uint8)*255
            for c in classification_mask_reject_list:
                mask_image[class_image==c]=0
    else:
        mask_image = np.ones_like(reference_image,dtype=np.uint8)


    benchmark_result = compute_benchmark(reference_image,
                                         test_image,
                                         mask_image = mask_image,
                                         z_tolerance = z_tolerance,
                                         show_figures = show_figures
                                        )

    print('completeness =', benchmark_result[4])
    result = test_info +  list(benchmark_result) + list(transformation[0]) + test_angles

    if use_classification_mask:
        result.append(classification_mask_accept_list)
        result.append(classification_mask_reject_list)
        results_column_names += ['mask_accepted_classes', 'mask_rejected_classes']

    results.append(result)


print(results)

#--------------------------------------------------------------------------------------------

timestr = time.strftime("%Y%m%d-%H%M%S")

if registration_algorithm=='nccshift':
    regstr = '{}_pixel_range_{}_bdint_{}_{}_dz_{}'.format(registration_algorithm, registration_pixel_range,
                                                                bdint_interpolate_moving_image, bdint_min_max_avg_option,
                                                               adjust_dz_method)
elif registration_algorithm=='pcc':
    regstr = '{}'.format(registration_algorithm)
elif registration_algorithm=='fixed':
    transfstr = '_'.join([str(a) for a in fixed_transformation[0]])
    regstr = '{}_{}'.format(registration_algorithm, transfstr)

if use_classification_mask:
    maskstr_accept = '_'.join([str(a) for a in classification_mask_accept_list])
    maskstr_reject = '_'.join([str(a) for a in classification_mask_reject_list])
    maskstr = 'mask_accept_' + maskstr_accept
    maskstr += '_mask_reject_' + maskstr_reject
else:
    maskstr = 'no'


results_csv_filename = os.path.join(base_directory, '{}_{}_{}_reg_{}_mask_{}_results_{}.csv'.format(title, prefix, nit_or_fit,
                                                                                                 regstr, maskstr,
                                                                                                 timestr))
#results_csv_filename = os.path.join('.', 'results_{}.csv'.format(timestr))


header = ";".join([str(x) for x in results_column_names])
print(header)


np.savetxt(results_csv_filename, results, header=header,comments='', delimiter=";", fmt="%s")
