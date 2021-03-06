import fnmatch
from glob import glob
import numpy as np
import imutils
import csv
import cv2
import os

import time

time_start = time.time()

bibs_path        = "data/number_recognition/annotations/bibs"
annotations_path = "data/number_recognition/annotations/bibs.csv"
annotations = {}

saved_annotations = []

def save_annotation(image_path, number):
    if image_path in saved_annotations:
        return

    saved_annotations.append(image_path)
    with open(annotations_path, 'a') as csv_file:
        if number == "":
            csv_file.write("%s,n,\n" % image_path)
        else:
            csv_file.write("%s,y,%s\n" % (image_path, number))

def save_partial_annotations():
    limit = len(annotations) - 5
    for idx, annotation in enumerate(annotations.items()):
        if idx >= limit:
            break

        image_path, number = annotation
        save_annotation(image_path, number)

def save_annotations():
    print("-" * 80)
    print("Saved " + str(len(annotations)) + " annotations: " +
        str(len(saved_annotations)) + " during classification, " +
        str(len(annotations) - len(saved_annotations)) + " at close")
    for image_path, number in annotations.items():
        save_annotation(image_path, number)

def annotate(image_path):
    print("")
    print(image_path)

    bib = cv2.imread(image_path)
    bib = cv2.resize(bib, (120, 120))
    cv2.imshow("Bib", bib)

    number = ""
    while True:
        key = cv2.waitKey(0)
        if key == 46 or key == 43 or key == 45 or key == 97: # chars '.', '+' & '-'
            break
        elif key >= 48 and key <= 57: # digit
            digit = key - 48
            number = number + str(digit)
        elif key == 13:
            break
        elif key == 8:
            return False
        elif key == 113: # 'q'
            save_annotations()
            exit(0)
        else:
            print("Invalid key: %d" % key)

    print("Number: " + number)
    annotations[image_path] = number

    return True

annotated_files = []
with open(annotations_path, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")
    for annotation in csv_reader:
        annotated_files.append(annotation[0])

def traverse_path(root, pattern):
    results = []
    for base, directories, files in os.walk(root):
        for result in fnmatch.filter(files, pattern):
            result = os.path.join(base, result)
            results.append(result)
        for directory in directories:
            for result in traverse_path(os.path.join(root, directory), pattern):
                results.append(result)

    return results
image_paths = traverse_path(bibs_path, "*.jpg")

index = 0
count = 0
while index < len(image_paths) and count < 500:
    image_path = image_paths[index]
    if image_path in annotated_files:
        print("Already annotated: " + image_path)
        index = index + 1
        continue

    if annotate(image_path):
        index = index + 1
        count = count + 1
        save_partial_annotations()
    else:
        index = index - 1
        count = count - 1

save_annotations()

elapsed_seconds = int(time.time() - time_start)
elapsed_minutes = elapsed_seconds / 60
elapsed_seconds = elapsed_seconds % 60
print("Time spent: %dm %ds" % (elapsed_minutes, elapsed_seconds))
