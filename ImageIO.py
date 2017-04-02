#!/usr/bin/env python
# -*- coding=utf-8 -*-

# This is the only file containing PIL/scipy calls

import os
import sys

try:
    from scipy import misc
except ImportError:
    print "Scipy import error."
    sys.exit(-0x2)


def load_image(filename):
    if not os.path.exists(filename):
        print "No such file or directory %s" % filename
        sys.exit(-0x3)
    return misc.imread(filename)


def save_image(filename, arr):
    misc.imsave(filename, arr)
