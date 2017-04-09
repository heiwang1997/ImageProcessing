#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import sys

import ImageIO
import numpy as np
import argparse


def change_brightness(arr, amount):
    output = np.maximum(np.minimum(arr.astype(int) + amount, 255), 0)
    return output.astype(np.uint8)


def change_contrast(arr, amount):
    trc = (arr - 127.0) * amount + 127.0
    output = np.maximum(np.minimum(trc + amount, 255), 0)
    return output.astype(np.uint8)


def change_gamma(arr, amount):
    output = 255 * ((arr / 255.0) ** (1 / amount))
    return output.astype(np.uint8)


def change_saturation(arr, amount):
    output = np.ndarray(arr.shape, dtype=np.uint8)
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            [red, green, blue] = arr[r, c, :]
            rgb_max = int(max(red, green, blue))
            rgb_min = int(min(red, green, blue))
            if rgb_max == rgb_min:
                output[r, c, :] = arr[r, c, :]
                continue
            delta = (rgb_max - rgb_min) / 255.0
            value = (rgb_max + rgb_min) / 255.0
            L = value / 2
            if L < 0.5:
                S = delta / value
            else:
                S = delta / (2.0 - value)
            if amount >= 0:
                if amount + S >= 1:
                    alpha = S
                else:
                    alpha = 1 - amount
                alpha = 1.0 / alpha - 1
                red = red + (red - L * 255) * alpha
                green = green + (green - L * 255) * alpha
                blue = blue + (blue - L * 255) * alpha
            else:
                red = L * 255 + (red - L * 255) * (1 + amount)
                green = L * 255 + (green - L * 255) * (1 + amount)
                blue = L * 255 + (blue - L * 255) * (1 + amount)
            output[r, c, :] = [red, green, blue]
    return output


def generate_histogram(arr):
    histogram = [0] * 256
    unique, counts = np.unique(arr, return_counts=True)
    for val, count in zip(unique, counts):
        histogram[val] = count
    return histogram


def get_pdf(arr):
    histo = generate_histogram(arr)
    for i in range(1, 256):
        histo[i] += histo[i - 1]
    max_value = histo[-1]
    return [float(t) / max_value for t in histo]


def histogram_equalization(arr):
    output = np.ndarray(arr.shape, dtype=np.uint8)
    for i in range(3):
        slice_band = arr[:, :, i]
        slice_pdf = get_pdf(slice_band)
        for x in np.nditer(slice_band, op_flags=['readwrite']):
            x[...] = np.uint8(255 * slice_pdf[x])
        output[:, :, i] = slice_band
    return output


def histogram_matching(arr, target):
    output = np.ndarray(arr.shape, dtype=np.uint8)
    for i in range(3):
        slice_band = arr[:, :, i]
        target_band = target[:, :, i]
        slice_pdf = get_pdf(slice_band)
        target_pdf = get_pdf(target_band)
        lut = [0] * 256
        gj = 0
        for gi in range(256):
            while target_pdf[gj] < slice_pdf[gi] and gj < 255:
                gj += 1
            lut[gi] = gj
        for x in np.nditer(slice_band, op_flags=['readwrite']):
            x[...] = np.uint8(lut[x])
        output[:, :, i] = slice_band
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mandatory fields
    parser.add_argument("filename", help="The filename of the input picture")
    parser.add_argument("output", help="The filename of the output picture")
    # Available parameters
    parser.add_argument("--brightness", metavar='amount', type=int)
    parser.add_argument("--contrast", metavar='amount', type=float)
    parser.add_argument("--gamma", metavar='amount', type=float)
    parser.add_argument("--histogramEQ", type=bool)
    parser.add_argument("--histogramMatch", type=str)
    parser.add_argument("--saturation", metavar='amount', type=float, help="From -1 to 1")
    args = parser.parse_args()

    img_arr = ImageIO.load_image(args.filename)
    if args.brightness:
        img_arr = change_brightness(img_arr, args.brightness)
    if args.contrast:
        img_arr = change_contrast(img_arr, args.contrast)
    if args.gamma:
        img_arr = change_gamma(img_arr, args.gamma)
    if args.histogramEQ:
        img_arr = histogram_equalization(img_arr)
    if args.histogramMatch:
        target_arr = ImageIO.load_image(args.histogramMatch)
        img_arr = histogram_matching(img_arr, target_arr)
    if args.saturation:
        img_arr = change_saturation(img_arr, args.saturation)

    ImageIO.save_image(args.output, img_arr)
