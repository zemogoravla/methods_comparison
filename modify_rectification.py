import argparse
import s2p
from skimage.io import imread, imsave
from skimage.transform import warp, warp_coords, AffineTransform
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# image normalization as done by VisSatSatelliteStereo
def normalize_image(I, gamma=1/2.2, lower_percentile=2 , upper_percentile=98):
    J = I.copy().astype(np.double)
    J = J**gamma
    lower = np.nanpercentile(J, lower_percentile)
    upper = np.nanpercentile(J, upper_percentile)
    J = (J - lower) / (upper - lower)
    J[J < 0] = 0
    J[J > 1] = 1
    return (J * 255).astype('uint8')


def get_matches_from_masked_disp(disp, mask):
    x = np.arange(disp.shape[1])
    y = np.arange(disp.shape[0])
    xv, yv = np.meshgrid(x, y,indexing='xy')
    xx = xv.ravel()
    yy = yv.ravel()
    dd = disp.ravel()
    valid_indices = np.logical_and(mask[yy,xx]>0, ~ np.isnan(dd))

    matches = np.vstack((xx[valid_indices],
                         yy[valid_indices],
                         xx[valid_indices] + dd[valid_indices],
                         yy[valid_indices]))

    matches = matches.T

    return matches


def modify_rectification(input_im1_filename, input_im2_filename, input_disp_filename, input_mask_filename,
                         output_im1_filename, output_im2_filename, output_disp_filename, output_mask_filename,
                         precomputed_matches_filename=None,
                         register_horizontally_shear=True,
                         register_horizontally_translation=True,
                         register_horizontally_translation_flag='negative',
                         debug=False):

    def debug_print(*args):
        if debug:
            print(args)

    im1 = imread(input_im1_filename)
    im2 = imread(input_im2_filename)
    disp = imread(input_disp_filename)
    mask = imread(input_mask_filename)

    if not precomputed_matches_filename is None:
        matches = np.loadtxt(precomputed_matches_filename)
    else:
        matches = get_matches_from_masked_disp(disp, mask)

    H1 = np.eye(3)
    H2 = np.eye(3)

    if register_horizontally_shear:
        debug_print('register_horizontally_shear--------------------------------')
        H2 = s2p.rectification.register_horizontally_shear(matches, H1, H2)
        debug_print('H2\n', H2)
        debug_print('register_horizontally_translation')
        H2 = s2p.rectification.register_horizontally_translation(matches, H1, H2, flag='negative')
        debug_print('H2\n', H2)

    H1 = np.eye(3)
    H2 = np.eye(3)

    if register_horizontally_shear:
        debug_print('register_horizontally_shear ---------------------------------')
        H2 = s2p.rectification.register_horizontally_shear(matches, H1, H2)
        debug_print('H2\n', H2)

    if register_horizontally_translation:
        debug_print('register_horizontally_translation----------------------------')
        H2 = s2p.rectification.register_horizontally_translation(matches, H1, H2, flag=register_horizontally_translation_flag)
        debug_print('H2\n', H2)


    debug_print('disparity_range_from_matches-------------------------------------')
    x = 0;
    y = 0
    w = im1.shape[1];
    h = im1.shape[0]
    disp_m, disp_M = s2p.rectification.disparity_range_from_matches(matches, H1, H2, w, h)
    debug_print('disp_m, disp_M', disp_m, disp_M)

    # recompute hmargin and homographies
    debug_print('recompute hmargin and homographies--------------------------------')
    hmargin = 0;
    vmargin = 0
    hmargin = int(np.ceil(max([hmargin, np.fabs(disp_m), np.fabs(disp_M)])))
    T = s2p.common.matrix_translation(hmargin, vmargin)
    H1, H2 = np.dot(T, H1), np.dot(T, H2)
    debug_print('H1\n', H1)
    debug_print('H2\n', H2)

    roi = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    pts1 = s2p.common.points_apply_homography(H1, roi)
    x0, y0, w0, h0 = s2p.common.bounding_box2D(pts1)

    debug_print('apply homographies and do the crops--------------------------------')
    # apply homographies and do the crops
    w = int(w0 + 2 * hmargin)
    h = int(h0 + 2 * vmargin)

    # # s2p warping alternative
    # s2p.common.image_apply_homography(output_im1_filename, input_im1_filename, H1, w, h)
    # s2p.common.image_apply_homography(output_im2_filename, input_im2_filename, H2, w, h)
    # s2p.common.image_apply_homography(output_mask_filename, input_mask_filename, H1, w, h)

    im1 = imread(input_im1_filename)
    im2 = imread(input_im2_filename)
    mask = imread(input_mask_filename)

    warped_im1 = warp(im1, inverse_map=AffineTransform(H1).inverse, order=3, mode='constant', cval=0.0,
                      output_shape=(h, w))
    warped_im2 = warp(im2, inverse_map=AffineTransform(H2).inverse, order=3, mode='constant', cval=0.0,
                      output_shape=(h, w))
    warped_mask = warp(mask, inverse_map=AffineTransform(H1).inverse, order=1, mode='constant', cval=0.0,
                      output_shape=(h, w)).astype(np.uint8)  # 0,1 values
    #scale 1s to 255
    warped_mask[warped_mask>0]=255

    imsave(output_im1_filename, warped_im1.astype(np.float32))
    imsave(output_im2_filename, warped_im2.astype(np.float32))
    imsave(output_mask_filename, warped_mask.astype(np.uint8))

    debug_print('compute the new disparity------------------------------------------')
    D = disp  # solo un nombre conveniente

    D_no_nans = D.copy()
    D_no_nans[np.isnan(D)] = 1e9  # use big number instead of nans. map_coordinates does not deal with nans

    # coordinates in the original disparity
    H1_transform = AffineTransform(H1)
    coords = warp_coords(H1_transform.inverse, (h, w))

    #
    t1 = H1[0, 2]
    a2 = H2[0, 0]
    b2 = H2[0, 1]
    t2 = H2[0, 2]

    # new disparity
    D_prime = a2 * map_coordinates(D_no_nans, coords, cval=np.nan) + (a2 - 1) * coords[1, :, :] + b2 * coords[0, :, :] + t2 - t1

    D_prime[D_prime > 1e4] = np.nan   # return big numbers to nans
    imsave(output_disp_filename, D_prime.astype(np.float32))



    #chequeos ------------------------------------------
    if debug:
        new_matches = matches.copy()
        new_matches[:, 0] += t1
        new_matches[:, 2] = new_matches[:, 2] * a2 + new_matches[:, 3] * b2 + t2
        # disparity for original matches
        d = D[matches[:,1].astype(int), matches[:,0].astype(int)]
        # disparity for new matches
        d_prime = D_prime[new_matches[:,1].astype(int), new_matches[:,0].astype(int)]
        d_prime_from_matches = new_matches[:,2]-new_matches[:,0]
        d_prime_diff = d_prime_from_matches - d_prime
        debug_print('min_diff={}, max_diff={}'.format(np.min(d_prime_diff), np.max(d_prime_diff)))
        plt.figure()
        plt.hist(d_prime_diff)
        plt.title('Histogram of differences (distance_between_new_matches - new_disp)')
        plt.show()

        #show some matches
        sample_count = 20
        sample_indices = np.random.choice( np.arange(matches.shape[0]), sample_count, replace=False)
        im1 = imread(input_im1_filename)
        im2 = imread(input_im2_filename)
        disp = imread(input_disp_filename)
        plt.figure()
        plt.imshow(normalize_image(im1), cmap='gray')
        plt.plot(matches[sample_indices,0], matches[sample_indices,1], '+r')
        plt.title('im1')
        plt.figure()
        plt.imshow(normalize_image(im2), cmap='gray')
        plt.plot(matches[sample_indices, 2], matches[sample_indices, 3], '+r')
        plt.title('im2')
        plt.figure()
        plt.imshow(disp, cmap='terrain')
        plt.plot(matches[sample_indices, 0], matches[sample_indices, 1], '+r')
        plt.title('disp')

        new_im1 = imread(output_im1_filename)
        new_im2 = imread(output_im2_filename)
        new_disp = imread(output_disp_filename)
        plt.figure()
        plt.imshow(normalize_image(new_im1), cmap='gray')
        plt.plot(new_matches[sample_indices, 0], new_matches[sample_indices, 1], '+r')
        plt.title('new im1')
        plt.figure()
        plt.imshow(normalize_image(new_im2), cmap='gray')
        plt.plot(new_matches[sample_indices, 2], new_matches[sample_indices, 3], '+r')
        plt.title('new im2')
        plt.figure()
        plt.imshow(new_disp, cmap='terrain')
        plt.plot(new_matches[sample_indices, 0], new_matches[sample_indices, 1], '+r')
        plt.title('new disp')

        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Modify a rectified pair')
    parser.add_argument('input_im1_filename', type=str)
    parser.add_argument('input_im2_filename', type=str)
    parser.add_argument('input_disp_filename', type=str)
    parser.add_argument('input_mask_filename', type=str)
    parser.add_argument('output_im1_filename', type=str)
    parser.add_argument('output_im2_filename', type=str)
    parser.add_argument('output_disp_filename', type=str)
    parser.add_argument('output_mask_filename', type=str)

    parser.add_argument('--register_horizontally_shear', type=bool, default=True)
    parser.add_argument('--register_horizontally_translation', type=bool, default=True)
    parser.add_argument('--register_horizontally_translation_flag', type=str, default='negative')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    modify_rectification(args.input_im1_filename, args.input_im2_filename, args.input_disp_filename, args.input_mask_filename,
                         args.output_im1_filename, args.output_im2_filename, args.output_disp_filename, args.output_mask_filename,
                         register_horizontally_shear=args.register_horizontally_shear,
                         register_horizontally_translation=args.register_horizontally_translation,
                         register_horizontally_translation_flag=args.register_horizontally_translation_flag,
                         debug=args.debug
                         )

