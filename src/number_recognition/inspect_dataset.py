import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))

from glob import glob
import re
import csv

import numpy as np
import imutils
import cv2

import config

def inspect(image_path, number):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (80, 80))
    footer = np.zeros((40, 80, 3), np.uint8)
    cv2.putText(footer, number, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    image = np.vstack([image, footer])

    cv2.imshow("Bib", image)
    key = cv2.waitKey(0)

def inspect_dataset(annotations_path):
    with open(annotations_path, "r") as csv_file:
        annotation_reader = csv.reader(csv_file, delimiter = ",")
        for annotation in annotation_reader:
            image_path, _, number = annotation
            inspect(image_path, number)

if __name__ == '__main__':
    inspect_dataset(config.annotations_path)
