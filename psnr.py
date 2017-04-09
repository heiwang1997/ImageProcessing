#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import argparse

import math

import sys

import ImageIO
import numpy as np


def PSNR(original, compressed):
    rows = original.shape[0]
    cols = original.shape[1]
    bands = original.shape[2]
    mse = np.sum(np.abs(original - compressed) ** 2)
    mse /= (rows * cols * bands)
    max_i = 255
    return 10 * np.log10(max_i * max_i / mse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mandatory fields
    parser.add_argument("filename1", help="The filename of the original picture")
    parser.add_argument("filename2", help="The filename of the manipulated picture")
    args = parser.parse_args()

    img_arr1 = ImageIO.load_image(args.filename1)
    img_arr2 = ImageIO.load_image(args.filename2)
    if img_arr1.shape != img_arr2.shape:
        print("Cannot calculate PSNR between two images with different sizes!")
        sys.exit(-3)
    print(PSNR(img_arr1, img_arr2))