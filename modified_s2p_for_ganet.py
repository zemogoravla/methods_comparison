import os
import shutil
import argparse
import datetime
import multiprocessing
import collections
import rasterio
import numpy as np

import s2p
from s2p.block_matching import rectify_secondary_tile_only, create_rejection_mask

from s2p.config import cfg
from s2p import common
from s2p import parallel
from s2p import initialization
from s2p import pointing_accuracy
from s2p import rectification
from s2p import block_matching
from s2p import masking
from s2p import triangulation
from s2p import fusion
from s2p import rasterization
from s2p import visualisation
from s2p import ply


from s2p import global_pointing_correction
from s2p import pointing_correction
#from s2p import rectification_pair
from s2p import disparity_to_height
from s2p import mean_heights
from s2p import global_mean_heights
from s2p import heights_to_ply
from s2p import disparity_to_ply
from s2p import plys_to_dsm
from s2p import global_dsm
#from s2p import stereo_matching


from s2p import rpc_utils
from s2p import estimation
from s2p import evaluation
from s2p import common
from s2p import visualisation
from s2p import block_matching
from s2p.config import cfg


def rectification_pair(tile, i):
    """
    Rectify a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y, w, h = tile['coordinates']
    img1 = cfg['images'][0]['img']
    rpc1 = cfg['images'][0]['rpcm']
    img2 = cfg['images'][i]['img']
    rpc2 = cfg['images'][i]['rpcm']
    pointing = os.path.join(cfg['out_dir'],
                            'global_pointing_pair_{}.txt'.format(i))

    outputs = ['disp_min_max.txt', 'rectified_ref.tif', 'rectified_sec.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('rectification: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('rectifying tile {} {} pair {}...'.format(x, y, i))
    try:
        A = np.loadtxt(os.path.join(out_dir, 'pointing.txt'))
    except IOError:
        A = np.loadtxt(pointing)
    try:
        m = np.loadtxt(os.path.join(out_dir, 'sift_matches.txt'))
    except IOError:
        m = None

    x, y, w, h = tile['coordinates']

    cur_dir = os.path.join(tile['dir'],'pair_{}'.format(i))
    for n in tile['neighborhood_dirs']:
        nei_dir = os.path.join(tile['dir'], n, 'pair_{}'.format(i))
        if os.path.exists(nei_dir) and not os.path.samefile(cur_dir, nei_dir):
            sift_from_neighborhood = os.path.join(nei_dir, 'sift_matches.txt')
            try:
                m_n = np.loadtxt(sift_from_neighborhood)
                # added sifts in the ellipse of semi axes : (3*w/4, 3*h/4)
                m_n = m_n[np.where(np.linalg.norm([(m_n[:,0]-(x+w/2))/w,
                                                   (m_n[:,1]-(y+h/2))/h],
                                                  axis=0) < 3.0/4)]
                if m is None:
                    m = m_n
                else:
                    m = np.concatenate((m, m_n))
            except IOError:
                print('%s does not exist' % sift_from_neighborhood)

    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    H1, H2, disp_min, disp_max = rectify_pair(img1, img2,
                                                            rpc1, rpc2,
                                                            x, y, w, h,
                                                            rect1, rect2, A, m,
                                                            method=cfg['rectification_method'],
                                                            hmargin=cfg['horizontal_margin'],
                                                            vmargin=cfg['vertical_margin'])
    np.savetxt(os.path.join(out_dir, 'H_ref.txt'), H1, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'H_sec.txt'), H2, fmt='%12.6f')
    np.savetxt(os.path.join(out_dir, 'disp_min_max.txt'), [disp_min, disp_max],
                            fmt='%3.1f')

    if cfg['clean_intermediate']:
        common.remove(os.path.join(out_dir,'pointing.txt'))
        common.remove(os.path.join(out_dir,'sift_matches.txt'))


def rectify_pair(im1, im2, rpc1, rpc2, x, y, w, h, out1, out2, A=None, sift_matches=None,
                 method='rpc', hmargin=0, vmargin=0):
    """
    Rectify a ROI in a pair of images.

    Args:
        im1, im2: paths to two GeoTIFF image files
        rpc1, rpc2: two instances of the rpcm.RPCModel class
        x, y, w, h: four integers defining the rectangular ROI in the first
            image.  (x, y) is the top-left corner, and (w, h) are the dimensions
            of the rectangle.
        out1, out2: paths to the output rectified crops
        A (optional): 3x3 numpy array containing the pointing error correction
            for im2. This matrix is usually estimated with the pointing_accuracy
            module.
        sift_matches (optional): Nx4 numpy array containing a list of sift
            matches, in the full image coordinates frame
        method (default: 'rpc'): option to decide wether to use rpc of sift
            matches for the fundamental matrix estimation.
        {h,v}margin (optional): horizontal and vertical margins added on the
            sides of the rectified images

    Returns:
        H1, H2: Two 3x3 matrices representing the rectifying homographies that
        have been applied to the two original (large) images.
        disp_min, disp_max: horizontal disparity range
    """
    # compute real or virtual matches
    if method == 'rpc':
        # find virtual matches from RPC camera models
        matches = rpc_utils.matches_from_rpc(rpc1, rpc2, x, y, w, h,
                                             cfg['n_gcp_per_axis'])

        # correct second image coordinates with the pointing correction matrix
        if A is not None:
            matches[:, 2:] = common.points_apply_homography(np.linalg.inv(A),
                                                            matches[:, 2:])
    elif method == 'sift':
        matches = sift_matches

    else:
        raise Exception("Unknown value {} for argument 'method'".format(method))

    # compute rectifying homographies
    H1, H2, F = rectification.rectification_homographies(matches, x, y, w, h)

    if cfg['register_with_shear']:
        # compose H2 with a horizontal shear to reduce the disparity range
        a = np.mean(rpc_utils.altitude_range(rpc1, x, y, w, h))
        lon, lat, alt = rpc_utils.ground_control_points(rpc1, x, y, w, h, a, a, 4)
        x1, y1 = rpc1.projection(lon, lat, alt)[:2]
        x2, y2 = rpc2.projection(lon, lat, alt)[:2]
        m = np.vstack([x1, y1, x2, y2]).T
        m = np.vstack({tuple(row) for row in m})  # remove duplicates due to no alt range
        H2 = rectification.register_horizontally_shear(m, H1, H2)

    # compose H2 with a horizontal translation to center disp range around 0
    if sift_matches is not None:
        sift_matches = rectification.filter_matches_epipolar_constraint(F, sift_matches,
                                                          cfg['epipolar_thresh'])
        if len(sift_matches) < 10:
            print('WARNING: no registration with less than 10 matches')
        else:
            H2 = rectification.register_horizontally_translation(sift_matches, H1, H2, 'negative')  #AG

    # compute disparity range
    if cfg['debug']:
        out_dir = os.path.dirname(out1)
        np.savetxt(os.path.join(out_dir, 'sift_matches_disp.txt'),
                   sift_matches, fmt='%9.3f')
        visualisation.plot_matches(im1, im2, rpc1, rpc2, sift_matches, x, y, w, h,
                                   os.path.join(out_dir, 'sift_matches_disp.png'))
    disp_mM, H2 = disparity_range(rpc1, rpc2, x, y, w, h, H1, H2,
                                     sift_matches, A)
    disp_m, disp_M = disp_mM
    # recompute hmargin and homographies
    hmargin = int(np.ceil(max([hmargin, np.fabs(disp_m), np.fabs(disp_M)])))
    T = common.matrix_translation(hmargin, vmargin)
    H1, H2 = np.dot(T, H1), np.dot(T, H2)

    # compute rectifying homographies for non-epipolar mode (rectify the secondary tile only)
    if block_matching.rectify_secondary_tile_only(cfg['matching_algorithm']):
        H1_inv = np.linalg.inv(H1)
        H1 = np.eye(3) # H1 is replaced by 2-D array with ones on the diagonal and zeros elsewhere
        H2 = np.dot(H1_inv,H2)
        T = common.matrix_translation(-x + hmargin, -y + vmargin)
        H1 = np.dot(T, H1)
        H2 = np.dot(T, H2)

    # compute output images size
    roi = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
    pts1 = common.points_apply_homography(H1, roi)
    x0, y0, w0, h0 = common.bounding_box2D(pts1)
    # check that the first homography maps the ROI in the positive quadrant
    np.testing.assert_allclose(np.round([x0, y0]), [hmargin, vmargin], atol=.01)

    # apply homographies and do the crops
    common.image_apply_homography(out1, im1, H1, w0 + 2*hmargin, h0 + 2*vmargin)
    common.image_apply_homography(out2, im2, H2, w0 + 2*hmargin, h0 + 2*vmargin)

    if block_matching.rectify_secondary_tile_only(cfg['matching_algorithm']):
        pts_in = [[0, 0], [disp_m, 0], [disp_M, 0]]
        pts_out = common.points_apply_homography(H1_inv,
                                                 pts_in)
        disp_m = pts_out[1,:] - pts_out[0,:]
        disp_M = pts_out[2,:] - pts_out[0,:]

    return H1, H2, disp_m, disp_M

def disparity_range(rpc1, rpc2, x, y, w, h, H1, H2, matches, A=None):
    """
    Compute the disparity range of a ROI from a list of point matches.

    Args:
        rpc1, rpc2 (rpcm.RPCModel): two RPC camera models
        x, y, w, h (int): 4-tuple of integers defining the rectangular ROI in
            the first image. (x, y) is the top-left corner, and (w, h) are the
            dimensions of the rectangle.
        H1, H2 (np.array): two rectifying homographies, stored as 3x3 arrays
        matches (np.array): Nx4 array containing a list of sift matches, in the
            full image coordinates frame
        A (np.array): 3x3 array containing the pointing error correction for
            im2. This matrix is usually estimated with the pointing_accuracy
            module.

    Returns:
        disp: 2-uple containing the horizontal disparity range
    """
    # compute exogenous disparity range if needed
    if cfg['disp_range_method'] in ['exogenous', 'wider_sift_exogenous']:
        exogenous_disp = rpc_utils.exogenous_disp_range_estimation(rpc1, rpc2,
                                                                   x, y, w, h,
                                                                   H1, H2, A,
                                                                   cfg['disp_range_exogenous_high_margin'],
                                                                   cfg['disp_range_exogenous_low_margin'])

        print("exogenous disparity range:", exogenous_disp)

    # compute SIFT disparity range if needed
    if cfg['disp_range_method'] in ['sift', 'wider_sift_exogenous']:
        if matches is not None and len(matches) >= 2:
            sift_disp = rectification.disparity_range_from_matches(matches, H1, H2, w, h)
        else:
            sift_disp = None
        print("SIFT disparity range:", sift_disp)

    # compute altitude range disparity if needed
    if cfg['disp_range_method'] == 'fixed_altitude_range':
        #alt_disp = rpc_utils.altitude_range_to_disp_range(cfg['alt_min'],
        alt_min_disp, alt_max_disp, H2 = altitude_range_to_disp_range(cfg['alt_min'],
                                                          cfg['alt_max'],
                                                          rpc1, rpc2,
                                                          x, y, w, h, H1, H2, A)
        alt_disp = (alt_min_disp, alt_max_disp)
        print("disparity range computed from fixed altitude range:", alt_disp)

    # now compute disparity range according to selected method
    if cfg['disp_range_method'] == 'exogenous':
        disp = exogenous_disp

    elif cfg['disp_range_method'] == 'sift':
        disp = sift_disp

    elif cfg['disp_range_method'] == 'wider_sift_exogenous':
        if sift_disp is not None and exogenous_disp is not None:
            disp = min(exogenous_disp[0], sift_disp[0]), max(exogenous_disp[1], sift_disp[1])
        else:
            disp = sift_disp or exogenous_disp

    elif cfg['disp_range_method'] == 'fixed_altitude_range':
        disp = alt_disp


    elif cfg['disp_range_method'] == 'fixed_pixel_range':
        disp = cfg['disp_min'], cfg['disp_max']

    # default disparity range to return if everything else broke
    if disp is None:
        disp = -3, 3

    # impose a minimal disparity range (TODO this is valid only with the
    # 'center' flag for register_horizontally_translation)
    #AG202009     disp = min(-3, disp[0]), max(3, disp[1])

    print("Final disparity range:", disp)
    return disp, H2


def alt_to_disp(rpc1, rpc2, x, y, alt, H1, H2, A=None):
    """
    Converts an altitude into a disparity.

    Args:
        rpc1: an instance of the rpcm.RPCModel class for the reference
            image
        rpc2: an instance of the rpcm.RPCModel class for the secondary
            image
        x, y: coordinates of the point in the reference image
        alt: altitude above the WGS84 ellipsoid (in meters) of the point
        H1, H2: rectifying homographies
        A (optional): pointing correction matrix

    Returns:
        the horizontal disparity of the (x, y) point of im1, assuming that the
        3-space point associated has altitude alt. The disparity is made
        horizontal thanks to the two rectifying homographies H1 and H2.
    """
    xx, yy = rpc_utils.find_corresponding_point(rpc1, rpc2, x, y, alt)[0:2]
    p1 = np.vstack([x, y]).T
    p2 = np.vstack([xx, yy]).T

    if A is not None:
        print("rpc_utils.alt_to_disp: applying pointing error correction")
        # correct coordinates of points in im2, according to A
        p2 = common.points_apply_homography(np.linalg.inv(A), p2)

    p1 = common.points_apply_homography(H1, p1)
    p2 = common.points_apply_homography(H2, p2)

    #AG----------------------------------------------
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    # correct H2 with a translation  so the disparities are negative (inspired in registration.register_horizontally_translation
    t = np.max(x2 - x1)  +  3  # el 3 es para estar lejos del cero
    H2 = np.dot(common.matrix_translation(-t, 0), H2)
    #-------------------------------------------------

    # np.testing.assert_allclose(p1[:, 1], p2[:, 1], atol=0.1)
    disp = p2[:, 0] - p1[:, 0] - t
    return disp, H2

def altitude_range_to_disp_range(m, M, rpc1, rpc2, x, y, w, h, H1, H2, A=None,
                                 margin_top=0, margin_bottom=0):
    """
    Args:
        m: min altitude over the tile
        M: max altitude over the tile
        rpc1: instance of the rpcm.RPCModel class for the reference image
        rpc2: instance of the rpcm.RPCModel class for the secondary image
        x, y, w, h: four integers defining a rectangular region of interest
            (ROI) in the reference image. (x, y) is the top-left corner, and
            (w, h) are the dimensions of the rectangle.
        H1, H2: rectifying homographies
        A (optional): pointing correction matrix

    Returns:
        the min and max horizontal disparity observed on the 4 corners of the
        ROI with the min/max altitude assumptions given as parameters. The
        disparity is made horizontal thanks to the two rectifying homographies
        H1 and H2.
    """
    # build an array with vertices of the 3D ROI, obtained as {2D ROI} x [m, M]
    a = np.array([x, x,   x,   x, x+w, x+w, x+w, x+w])
    b = np.array([y, y, y+h, y+h,   y,   y, y+h, y+h])
    c = np.array([m, M,   m,   M,   m,   M,   m,   M])

    # compute the disparities of these 8 points
    d, H2 = alt_to_disp(rpc1, rpc2, a, b, c, H1, H2, A)

    # return min and max disparities
    return np.min(d), np.max(d), H2


def main(user_cfg):
    """
    Launch the s2p pipeline with the parameters given in a json file.

    Args:
        user_cfg: user config dictionary
    """
    common.print_elapsed_time.t0 = datetime.datetime.now()
    initialization.build_cfg(user_cfg)
    initialization.make_dirs()

    # multiprocessing setup
    nb_workers = multiprocessing.cpu_count()  # nb of available cores
    if cfg['max_processes'] is not None:
        nb_workers = cfg['max_processes']

    tw, th = initialization.adjust_tile_size()
    tiles_txt = os.path.join(cfg['out_dir'], 'tiles.txt')
    tiles = initialization.tiles_full_info(tw, th, tiles_txt, create_masks=True)
    if not tiles:
        print('ERROR: the ROI is not seen in two images or is totally masked.')
        return

    # initialisation: write the list of tilewise json files to outdir/tiles.txt
    with open(tiles_txt, 'w') as f:
        for t in tiles:
            print(t['json'], file=f)

    n = len(cfg['images'])
    tiles_pairs = [(t, i) for i in range(1, n) for t in tiles]

    # local-pointing step:
    print('correcting pointing locally...')
    parallel.launch_calls(pointing_correction, tiles_pairs, nb_workers)

    # global-pointing step:
    print('correcting pointing globally...')
    global_pointing_correction(tiles)
    common.print_elapsed_time()

    # rectification step:
    print('rectifying tiles...')
    parallel.launch_calls(rectification_pair, tiles_pairs, nb_workers)

    # matching step:
    print('running stereo matching...')
    parallel.launch_calls(stereo_matching, tiles_pairs, nb_workers)

    if n > 2:
        # disparity-to-height step:
        print('computing height maps...')
        parallel.launch_calls(disparity_to_height, tiles_pairs, nb_workers)

        print('computing local pairwise height offsets...')
        parallel.launch_calls(mean_heights, tiles, nb_workers)

        # global-mean-heights step:
        print('computing global pairwise height offsets...')
        global_mean_heights(tiles)

        # heights-to-ply step:
        print('merging height maps and computing point clouds...')
        parallel.launch_calls(heights_to_ply, tiles, nb_workers)
    else:
        # triangulation step:
        print('triangulating tiles...')
        parallel.launch_calls(disparity_to_ply, tiles, nb_workers)

    # local-dsm-rasterization step:
    print('computing DSM by tile...')
    parallel.launch_calls(plys_to_dsm, tiles, nb_workers)

    # global-dsm-rasterization step:
    print('computing global DSM...')
    global_dsm(tiles)
    common.print_elapsed_time()

    # cleanup
    common.garbage_cleanup()
    common.print_elapsed_time(since_first_call=True)

def stereo_matching(tile,i):
    """
    Compute the disparity of a pair of images on a given tile.

    Args:
        tile: dictionary containing the information needed to process a tile.
        i: index of the processed pair
    """
    out_dir = os.path.join(tile['dir'], 'pair_{}'.format(i))
    x, y = tile['coordinates'][:2]

    outputs = ['rectified_mask.png', 'rectified_disp.tif']

    if os.path.exists(os.path.join(out_dir, 'stderr.log')):
        print('disparity estimation: stderr.log exists')
        print('pair_{} not processed on tile {} {}'.format(i, x, y))
        return

    print('estimating disparity on tile {} {} pair {}...'.format(x, y, i))
    rect1 = os.path.join(out_dir, 'rectified_ref.tif')
    rect2 = os.path.join(out_dir, 'rectified_sec.tif')
    disp = os.path.join(out_dir, 'rectified_disp.tif')
    mask = os.path.join(out_dir, 'rectified_mask.png')
    disp_min, disp_max = np.loadtxt(os.path.join(out_dir, 'disp_min_max.txt'))

    compute_disparity_map(rect1, rect2, disp, mask,
                                         cfg['matching_algorithm'], disp_min,
                                         disp_max)

    # add margin around masked pixels
    masking.erosion(mask, mask, cfg['msk_erosion'])

    if cfg['clean_intermediate']:
        if len(cfg['images']) > 2:
            common.remove(rect1)
        common.remove(rect2)
        common.remove(os.path.join(out_dir,'disp_min_max.txt'))


def compute_disparity_map(im1, im2, disp, mask, algo, disp_min=None,
                          disp_max=None, timeout=cfg['mgm_timeout'], extra_params=''):
    """
    Runs a block-matching binary on a pair of stereo-rectified images.

    Args:
        im1, im2: rectified stereo pair
        disp: path to the output diparity map
        mask: path to the output rejection mask
        algo: string used to indicate the desired binary. Currently it can be
            one among 'hirschmuller02', 'hirschmuller08',
            'hirschmuller08_laplacian', 'hirschmuller08_cauchy', 'sgbm',
            'msmw', 'tvl1', 'mgm', 'mgm_multi' and 'micmac'
        disp_min : smallest disparity to consider
        disp_max : biggest disparity to consider
        timeout: time in seconds after which the disparity command will
            raise an error if it hasn't returned.
            Only applies to `mgm*` algorithms.
        extra_params: optional string with algorithm-dependent parameters
    """
    if rectify_secondary_tile_only(algo) is False:
        disp_min = [disp_min]
        disp_max = [disp_max]

    # limit disparity bounds
    np.alltrue(len(disp_min) == len(disp_max))
    for dim in range(len(disp_min)):
        if disp_min[dim] is not None and disp_max[dim] is not None:
            image_size = common.image_size_gdal(im1)
            if disp_max[dim] - disp_min[dim] > image_size[dim]:
                center = 0.5 * (disp_min[dim] + disp_max[dim])
                disp_min[dim] = int(center - 0.5 * image_size[dim])
                disp_max[dim] = int(center + 0.5 * image_size[dim])

        # round disparity bounds
        if disp_min[dim] is not None:
            disp_min[dim] = int(np.floor(disp_min[dim]))
        if disp_max is not None:
            disp_max[dim] = int(np.ceil(disp_max[dim]))

    if rectify_secondary_tile_only(algo) is False:
        disp_min = disp_min[0]
        disp_max = disp_max[0]

    # define environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(cfg['omp_num_threads'])

    # call the block_matching binary
    if algo == 'hirschmuller02':
        bm_binary = 'subpix.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: LoG(0) regionRadius(3)
        #    LoG: Laplacian of Gaussian preprocess 1:enabled 0:disabled
        #    regionRadius: radius of the window

    if algo == 'hirschmuller08':
        bm_binary = 'callSGBM.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: regionRadius(3) P1(default) P2(default) LRdiff(1)
        #    regionRadius: radius of the window
        #    P1, P2 : regularization parameters
        #    LRdiff: maximum difference between left and right disparity maps

    if algo == 'hirschmuller08_laplacian':
        bm_binary = 'callSGBM_lap.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'hirschmuller08_cauchy':
        bm_binary = 'callSGBM_cauchy.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'sgbm':
        # opencv sgbm function implements a modified version of Hirschmuller's
        # Semi-Global Matching (SGM) algorithm described in "Stereo Processing
        # by Semiglobal Matching and Mutual Information", PAMI, 2008

        p1 = 8  # penalizes disparity changes of 1 between neighbor pixels
        p2 = 32  # penalizes disparity changes of more than 1
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity

        win = 3  # matched block size. It must be a positive odd number
        lr = 1  # maximum difference allowed in the left-right disparity check
        cost = common.tmpfile('.tif')
        common.run('sgbm {} {} {} {} {} {} {} {} {} {}'.format(im1, im2,
                                                               disp, cost,
                                                               disp_min,
                                                               disp_max,
                                                               win, p1, p2, lr))

        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'tvl1':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4}'.format(tvl1, im1, im2, disp, mask),
                   env)

    if algo == 'tvl1_2d':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4} {5}'.format(tvl1, im1, im2, disp, mask,
                                                    1), env)


    if algo == 'msmw':
        bm_binary = 'iip_stereo_correlation_multi_win2'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o 0.25 -f 0 -P 32 -m {1} -M {2} {3} {4} {5} {6}'.format(bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'msmw2':
        bm_binary = 'iip_stereo_correlation_multi_win2_newversion'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o -0.25 -f 0 -P 32 -D 0 -O 25 -c 0 -m {1} -M {2} {3} {4} {5} {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask), env)

    if algo == 'msmw3':
        bm_binary = 'msmw'
        common.run('{0} -m {1} -M {2} -il {3} -ir {4} -dl {5} -kl {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'mgm':
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['TSGM'] = '3'

        nb_dir = cfg['mgm_nb_directions']

        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi_lsd':


        ref = im1
        sec = im2


        wref = common.tmpfile('.tif')
        wsec = common.tmpfile('.tif')
        # TODO TUNE LSD PARAMETERS TO HANDLE DIRECTLY 12 bits images?
        # image dependent weights based on lsd segments
        image_size = common.image_size_gdal(ref)
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s'%(ref,image_size[0], image_size[1],wref),
                   shell=True)
        # image dependent weights based on lsd segments
        image_size = common.image_size_gdal(sec)
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s'%(sec,image_size[0], image_size[1],wsec),
                   shell=True)


        env['REMOVESMALLCC'] = str(cfg['stereo_speckle_filter'])
        env['SUBPIX'] = '2'
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        # increasing these numbers compensates the loss of regularity after incorporating LSD weights
        P1 = 12*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 48*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = disp+'.confidence.tif'

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-wl {wref} -wr {wsec} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                wref=wref,
                wsec=wsec,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'ganet':
        import test_ganet


        #ganet.test_ganet(im1, im2, disp, do_mismatch_filtering=True, stereo_speckle_filter=cfg['stereo_speckle_filter'])
        #ganet.test_ganet(im1, im2, disp, do_mismatch_filtering=True, stereo_speckle_filter=50)

        test_ganet.main(['--ref', im1, '--sec', im2, '--disp', disp,
                         '--pretrained', os.getenv('GANET_PRETRAINED_MODEL', '') ,
                         '--do_mismatch_filtering',
                         '--stereo_speckle_filter', str(cfg['stereo_speckle_filter']),
                         '--crop_height', os.getenv('GANET_CROP_HEIGHT', str(48*14)),
                         '--crop_width', os.getenv('GANET_CROP_WIDTH', str(48 * 14)),
                         '--max_disp', os.getenv('GANET_MAX_DISP', str(192)) ])


        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'mgm_multi':
        env['REMOVESMALLCC'] = str(cfg['stereo_speckle_filter'])
        env['MINDIFF'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['SUBPIX'] = '2'
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)

    if (algo == 'micmac'):
        # add micmac binaries to the PATH environment variable
        s2p_dir = os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
        micmac_bin = os.path.join(s2p_dir, 'bin', 'micmac', 'bin')
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + micmac_bin

        # prepare micmac xml params file
        micmac_params = os.path.join(s2p_dir, '3rdparty', 'micmac_params.xml')
        work_dir = os.path.dirname(os.path.abspath(im1))
        common.run('cp {0} {1}'.format(micmac_params, work_dir))

        # run MICMAC
        common.run('MICMAC {0:s}'.format(os.path.join(work_dir, 'micmac_params.xml')))

        # copy output disp map
        micmac_disp = os.path.join(work_dir, 'MEC-EPI',
                                   'Px1_Num6_DeZoom1_LeChantier.tif')
        disp = os.path.join(work_dir, 'rectified_disp.tif')
        common.run('cp {0} {1}'.format(micmac_disp, disp))

        # compute mask by rejecting the 10% of pixels with lowest correlation score
        micmac_cost = os.path.join(work_dir, 'MEC-EPI',
                                   'Correl_LeChantier_Num_5.tif')
        mask = os.path.join(work_dir, 'rectified_mask.png')
        common.run(["plambda", micmac_cost, "x x%q10 < 0 255 if", "-o", mask])



if __name__ == '__main__':

    """
    Command line parsing for s2p command line interface.
    """
    parser = argparse.ArgumentParser(description=('S2P: Satellite Stereo '
                                                  'Pipeline'))
    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing the paths to '
                              'input and output files and the algorithm '
                              'parameters'))
    args = parser.parse_args()

    user_cfg = s2p.read_config_file(args.config)

    user_cfg["matching_algorithm"]='ganet'

    # user_cfg['disp_range_method'] = 'fixed_altitude_range'
    # user_cfg['alt_min'] = 12
    # user_cfg['alt_max'] = 60
    # user_cfg['debug'] = True


    main(user_cfg)

    # Backup input file for sanity check
    if not args.config.startswith(os.path.abspath(s2p.cfg['out_dir']+os.sep)):
        shutil.copy2(args.config,os.path.join(s2p.cfg['out_dir'],'config.json.orig'))