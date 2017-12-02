import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))

from glob import glob
import re

import numpy as np
import imutils
import cv2

import config

def analyze_image(image_path):
    image = cv2.imread(image_path)
    width, height, depth = image.shape
    ratio = width / float(height)

    print("%s,%d,%d,%.2f" % (
        image_path,
        width, height,
        ratio))

    return (width, height, ratio)

def show_dataset_stats():
    image_path = []
    width_min = 40
    width_max = 120
    height_min = 40
    height_max = 120

    width_sum, height_sum, ratio_sum = 0, 0, 0
    w_count, h_count = 0, 0
    count = 0

    for image_path in glob(os.path.join(config.bibs_path, "**/*.jpg"), recursive=True):
        width, height, ratio = analyze_image(image_path)
        width_sum += width
        height_sum += height
        ratio_sum += ratio
        count += 1
        if width > width_min and width < width_max:
            w_count += 1
        if height > height_min and height < height_max:
            h_count += 1
    for image_path in glob(os.path.join(config.bibs_path, "**/*.JPG"), recursive=True):
        width, height, ratio = analyze_image(image_path)
        width_sum += width
        height_sum += height
        ratio_sum += ratio
        count += 1
        if width > width_min and width < width_max:
            w_count += 1
        if height > height_min and height < height_max:
            h_count += 1

    print("Summary:")
    print("-" * 80)
    print("Width average : %.2f" % (width_sum / float(count)))
    print("Height average: %.2f" % (height_sum / float(count)))
    print("Ratio average : %.2f" % (ratio_sum / float(count)))
    print("")
    print("In width range [%d-%d] : %d (%.2f%%)" % (
        width_min, width_max,
        w_count,
        (100 * w_count / float(count))))
    print("In height range [%d-%d]: %d (%.2f%%)" % (
        height_min, height_max,
        h_count,
        (100 * h_count / float(count))))

if __name__ == '__main__':
    show_dataset_stats()
