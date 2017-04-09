#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import argparse

import math
import ImageIO
import numpy as np


def get_sampling_scale(ori_r, ori_c, tgt_r, tgt_c):
    # if ori_r > tgt_r:
    #     # Down-sampling
    #     s_r = ori_r / tgt_r
    # else:
    #     # Up-sampling
    #     s_r = (ori_r - 1) / tgt_r
    # if ori_c > tgt_c:
    #     s_c = ori_c / tgt_c
    # else:
    #     s_c = (ori_c - 1) / tgt_c
    s_r = (ori_r - 1) / (tgt_r - 1)
    s_c = (ori_c - 1) / (tgt_c - 1)
    return s_r, s_c


def nearest_neighbour_resampling(arr, target_r, target_c):
    origin_r = arr.shape[0]
    origin_c = arr.shape[1]
    # Maybe a better solution should be (o_r - 1) / (t_r - 1)
    s_r, s_c = get_sampling_scale(origin_r, origin_c, target_r, target_c)
    output = np.ndarray([target_r, target_c, 3], dtype=np.uint8)
    # Populate pixel in output
    for r in range(target_r):
        for c in range(target_c):
            rf = s_r * r
            cf = s_c * c
            output[r, c, :] = arr[round(rf), round(cf), :]
    return output


def bilinear_resampling(arr, target_r, target_c):
    origin_r = arr.shape[0]
    origin_c = arr.shape[1]
    arr_padded = np.ndarray([origin_r + 1, origin_c + 1, 3], dtype=np.uint8)
    arr_padded[0:origin_r, 0:origin_c, :] = arr
    arr_padded[origin_r, 0:origin_c, :] = arr[origin_r - 1, :, :]
    arr_padded[:, origin_c, :] = arr_padded[:, origin_c - 1, :]
    s_r, s_c = get_sampling_scale(origin_r, origin_c, target_r, target_c)

    output = np.ndarray([target_r, target_c, 3], dtype=np.uint8)
    # Populate pixel in output
    for r in range(target_r):
        for c in range(target_c):
            rf, cf = s_r * r, s_c * c
            rp, cp = math.floor(rf), math.floor(cf)
            dr, dc = rf - rp, cf - cp
            output[r, c, :] = arr_padded[rp, cp, :] * (1 - dr) * (1 - dc) + \
                              arr_padded[rp + 1, cp, :] * dr * (1 - dc) + \
                              arr_padded[rp, cp + 1, :] * (1 - dr) * dc + \
                              arr_padded[rp + 1, cp + 1, :] * dr * dc
    return output


def bicubic_resampling(arr, target_r, target_c):
    origin_r = arr.shape[0]
    origin_c = arr.shape[1]
    # Padding for missing values
    arr_padded = np.ndarray([origin_r + 3, origin_c + 3, 3], dtype=np.uint8)
    arr_padded[1:origin_r + 1, 1:origin_c + 1, :] = arr
    arr_padded[1:origin_r + 1, 0, :] = arr[:, 0, :]
    arr_padded[0, 0:origin_c + 1, :] = arr_padded[1, 0:origin_c + 1, :]
    arr_padded[origin_r + 1, :, :] = arr_padded[origin_r, :, :]
    arr_padded[origin_r + 2, :, :] = arr_padded[origin_r, :, :]
    arr_padded[:, origin_c + 1, :] = arr_padded[:, origin_c, :]
    arr_padded[:, origin_c + 2, :] = arr_padded[:, origin_c, :]
    arr_padded = arr_padded.astype(np.float)
    s_r, s_c = get_sampling_scale(origin_r, origin_c, target_r, target_c)

    # Initialize some constant values.
    fmat = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])

    def get_mid_mat(row, col, band):
        def fx(row_t, col_t):
            return (arr_padded[row_t + 1, col_t, band] - arr_padded[row_t - 1, col_t, band]) / 2

        def fy(row_t, col_t):
            return (arr_padded[row_t, col_t + 1, band] - arr_padded[row_t, col_t - 1, band]) / 2

        def fxy(row_t, col_t):
            return (arr_padded[row_t + 1, col_t + 1, band] - arr_padded[row_t + 1, col_t - 1, band] +
                    arr_padded[row_t - 1, col_t - 1, band] - arr_padded[row_t - 1, col_t + 1, band]) / 4

        return np.matrix([[arr_padded[row, col, band], arr_padded[row, col + 1, band], fy(row, col), fy(row, col + 1)],
                          [arr_padded[row + 1, col, band], arr_padded[row + 1, col + 1, band],
                           fy(row + 1, col), fy(row + 1, col + 1)],
                          [fx(row, col), fx(row, col + 1), fxy(row, col), fxy(row, col + 1)],
                          [fx(row + 1, col), fx(row + 1, col + 1), fxy(row + 1, col), fxy(row + 1, col + 1)]])

    output = np.ndarray([target_r, target_c, 3], dtype=np.uint8)
    # Populate pixel in output
    for r in range(target_r):
        for c in range(target_c):
            rf, cf = s_r * r, s_c * c
            rp, cp = math.floor(rf), math.floor(cf)
            dr, dc = rf - rp, cf - cp
            x_arr = np.matrix([1, dr, dr * dr, dr * dr * dr])
            y_arr = np.matrix([1, dc, dc * dc, dc * dc * dc]).T
            for i in range(3):
                temp = x_arr * (fmat * get_mid_mat(rp + 1, cp + 1, i) * fmat.T) * y_arr
                temp = min(max(temp, 0), 255)
                output[r, c, i] = temp
        print("\rPlease Wait: %d / %d" % (r, target_r), end='')
    return output


def lanczos_resampling(arr, target_r, target_c):
    """
    Lanczos Resampling method with a = 3
    :param arr: input image
    :param target_r: output image rows (height)
    :param target_c: output image columns (width)
    :return: output image
    """
    origin_r = arr.shape[0]
    origin_c = arr.shape[1]
    s_r, s_c = get_sampling_scale(origin_r, origin_c, target_r, target_c)
    # Padding 3 to outside of arr.
    arr_padded = np.pad(arr, ((2, 3), (2, 3), (0, 0)), 'edge')
    output = np.ndarray([target_r, target_c, 3], dtype=np.uint8)

    def lanczos_filter(x):
        if x == 0:
            return 1
        elif -3 <= x < 3:
            return (3 * np.sin(np.pi * x) * np.sin(np.pi * x / 3)) / (np.pi * np.pi * x * x)
        else:
            return 0

    # Populate pixel in output
    for r in range(target_r):
        for c in range(target_c):
            rf, cf = s_r * r, s_c * c
            rp, cp = math.floor(rf), math.floor(cf)
            # Grab 36 pixels.
            region = arr_padded[rp:(rp + 6), cp:(cp + 6), :]
            coefs = np.ndarray((6, 6), dtype=float)
            for r_t in range(-2, 4):
                for c_t in range(-2, 4):
                    coefs[r_t + 2, c_t + 2] = lanczos_filter(np.sqrt((rp + r_t - rf) ** 2 + (cp + c_t - cf) ** 2))
            coefs /= np.sum(coefs)
            this_color = np.zeros((3, ), dtype=float)
            for r_t in range(6):
                for c_t in range(6):
                    this_color += region[r_t, c_t, :] * coefs[r_t, c_t]
            this_color = np.minimum(np.maximum(this_color, 0), 255)
            output[r, c, :] = this_color
        print("\rPlease Wait: %d / %d" % (r, target_r), end='')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mandatory fields
    parser.add_argument("filename", help="The filename of the input picture")
    parser.add_argument("output", help="The filename of the output picture")
    parser.add_argument("method", type=str, help="nn, bi, cu or la")
    parser.add_argument("r", type=int, help="height")
    parser.add_argument("c", type=int, help="width")
    args = parser.parse_args()

    img_arr = ImageIO.load_image(args.filename)
    if args.method == "nn":
        img_arr = nearest_neighbour_resampling(img_arr, args.r, args.c)
    elif args.method == "bi":
        img_arr = bilinear_resampling(img_arr, args.r, args.c)
    elif args.method == "cu":
        img_arr = bicubic_resampling(img_arr, args.r, args.c)
    elif args.method == "la":
        img_arr = lanczos_resampling(img_arr, args.r, args.c)
    else:
        print("Resampling method not supported. Try nn, bi, cu or la.")

    ImageIO.save_image(args.output, img_arr)
