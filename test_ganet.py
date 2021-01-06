from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
import sys
import shutil
import os
import re
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm


from models.GANet_deep import GANet

# from dataloader.data import get_test_set
import numpy as np
import matplotlib.pyplot as plt



def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


def test_transform_original(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim == 3:
        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
    else:
        r = g = b = left[:, :]

    temp_data[0, :, :] = (r - np.nanmean(r[:])) / np.nanstd(r[:])
    temp_data[1, :, :] = (g - np.nanmean(g[:])) / np.nanstd(g[:])
    temp_data[2, :, :] = (b - np.nanmean(b[:])) / np.nanstd(b[:])
    if left.ndim == 3:
        r = right[:, :, 0]
        g = right[:, :, 1]
        b = right[:, :, 2]
    else:
        r = g = b = right[:, :]

    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.nanmean(r[:])) / np.nanstd(r[:])
    temp_data[4, :, :] = (g - np.nanmean(g[:])) / np.nanstd(g[:])
    temp_data[5, :, :] = (b - np.nanmean(b[:])) / np.nanstd(b[:])
    return temp_data


def test(leftname, rightname, savename, opt, model):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if opt.cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()

    # agregado a sugenrencia de GC
    del prediction
    torch.cuda.empty_cache()
    # -----------------------------

    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    # skimage.io.imsave(savename, (temp * 256).astype('uint16'))
    skimage.io.imsave(savename, temp)
    return temp


#def test_ganet(rectified_left_filename, rectified_right_filename  , rectified_disp_filename, do_mismatch_filtering=False, stereo_speckle_filter=25):
def test_ganet(opt, model):
    import skimage
    from skimage.io import imread, imsave
    from tools.tiling import tile_image, untile_image
    from libTP import stereo

    rectified_left_filename = opt.ref
    rectified_right_filename = opt.sec
    rectified_disp_filename = opt.disp
    do_mismatch_filtering = opt.do_mismatch_filtering
    stereo_speckle_filter = opt.stereo_speckle_filter


    leftname = 'tmp_rectified_left_image.tif'
    rightname = 'tmp_rectified_right_image.tif'
    leftname_flipped = 'tmp_rectified_left_image_flipped.tif'
    rightname_flipped = 'tmp_rectified_right_image_flipped.tif'
    savename = 'tmp_rectified_left_disp.tif'

    rectified_left = imread(rectified_left_filename)
    rectified_right = imread(rectified_right_filename)

    # COMPUTE LR DISPARITY (done by tiles for gpu restrictions)
    print('GANET: computing LR disparity')
    rectified_left_tiles, tile_origins = tile_image(rectified_left, opt.crop_width, opt.crop_height)
    rectified_right_tiles, tile_origins = tile_image(rectified_right, opt.crop_width, opt.crop_height)

    prediction_tiles = []
    for i in tqdm(range(len(rectified_left_tiles))):
        skimage.io.imsave(leftname, rectified_left_tiles[i])
        skimage.io.imsave(rightname, rectified_right_tiles[i])
        prediction = test(leftname, rightname, savename, opt, model)
        prediction_tiles.append(prediction)

    disp_lr = untile_image(prediction_tiles, tile_origins)

    if do_mismatch_filtering:
        # COMPUTE RL DISPARITY (done by tiles for gpu restrictions)
        print('GANET: computing RL disparity')

        rectified_left_flipped = np.fliplr(rectified_left)
        rectified_right_flipped = np.fliplr(rectified_right)

        rectified_left_tiles, tile_origins = tile_image(rectified_right_flipped, opt.crop_width, opt.crop_height)
        rectified_right_tiles, tile_origins = tile_image(rectified_left_flipped, opt.crop_width, opt.crop_height)

        prediction_tiles = []
        for i in tqdm(range(len(rectified_left_tiles))):
            skimage.io.imsave(leftname, rectified_left_tiles[i])
            skimage.io.imsave(rightname, rectified_right_tiles[i])
            prediction = test(leftname, rightname, savename, opt, model)
            prediction_tiles.append(prediction)

        disp_rl = untile_image(prediction_tiles, tile_origins)
        disp_lrs = stereo.mismatchFiltering(disp_lr, -np.fliplr(disp_rl), stereo_speckle_filter)

    else:

        disp_lrs = disp_lr


    # save filenames
    if do_mismatch_filtering:
        disp_filename_no_extension, extension = os.path.splitext(rectified_disp_filename)
        unfiltered_lr_filename = disp_filename_no_extension + '_lr' + extension
        unfiltered_rl_filename = disp_filename_no_extension + '_rl' + extension
        skimage.io.imsave(unfiltered_lr_filename, -disp_lr)
        skimage.io.imsave(unfiltered_rl_filename, -disp_rl)

    skimage.io.imsave(rectified_disp_filename, -disp_lrs)


# https://stackoverflow.com/questions/14500183/in-python-can-i-call-the-main-of-an-imported-module
def main(args):
    # las dimensiones tienen que ser multiplo de 48 y tiene que dar la memoria de la GPU
    crop_val_laptop_AG = 48 * 14

    parser = argparse.ArgumentParser(description='PyTorch GANet Example')
    parser.add_argument('--ref', type=str, required=True, help="reference image")
    parser.add_argument('--sec', type=str, required=True, help="secondary image")
    parser.add_argument('--disp', type=str, required=True, help="disparity output image")
    parser.add_argument('--pretrained', type=str, required=True, help="use this pretrained model")
    parser.add_argument('--do_mismatch_filtering', action='store_true', help='Do left-right mismatch filtering')
    parser.add_argument('--stereo_speckle_filter', type=int, default=25, help="speckle filter size")


    # parser.add_argument('--crop_height', type=int, required=True, help="crop height")
    # parser.add_argument('--crop_width', type=int, required=True, help="crop width")
    parser.add_argument('--crop_height', type=int, default=crop_val_laptop_AG, help="crop height, must be 48*? (default=48*14)")
    parser.add_argument('--crop_width', type=int, default=crop_val_laptop_AG, help="crop width, must be 48*? (default=48*14)")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')

    # parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
    # parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
    # # parser.add_argument('--data_path', type=str, required=True, help="data root")
    # # parser.add_argument('--test_list', type=str, required=True, help="training list")
    # parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
    # parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
    # parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")

    # args_str = '--crop_height {} --crop_width {} --resume {}'.format(crop_val, crop_val, model_filename)

    opt = parser.parse_args(args)
    print(opt)

    cuda = opt.cuda
    # cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # print('===> Loading datasets')
    # test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width], false, opt.kitti, opt.kitti2015)
    # testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model')
    model = GANet(opt.max_disp)

    if cuda:
        model = torch.nn.DataParallel(model).cuda()

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        else:
            msg = "=> no checkpoint found at '{}'".format(opt.pretrained)
            print(msg)
            raise ValueError(msg)


    #test_ganet(opt.ref, opt.sec, opt.disp, opt.do_mismatch_filtering, opt.stereo_speckle_filter)
    test_ganet(opt,model)



if __name__ == "__main__":
    main(sys.argv[1:])
