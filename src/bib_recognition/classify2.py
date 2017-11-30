import fnmatch
from glob import glob
import numpy as np
import imutils
import csv
import cv2
import os

bibs_path        = "data/bib_recognition/annotations/bibs"
annotations_path = "data/bib_recognition/annotations/bibs.csv"
annotations = {}

def save_annotations():
    print("-" * 80)
    print("Saving " + str(len(annotations)) + " annotations")
    for image_path, digit in annotations.items():
        with open(annotations_path, 'a') as csv_file:
            if digit == None:
                csv_file.write("%(image_path)s,n,\n" % locals())
            else:
                csv_file.write("%(image_path)s,y,%(digit)d\n" % locals())

def annotate(image_path):
    print(image_path)

    bib = cv2.imread(image_path)
    cv2.imshow("Bib", bib)

    number = 0
    while True:
        key = cv2.waitKey(0)
        if key == 46 or key == 43 or key == 45 or key == 97: # chars '.', '+' & '-'
            number = None
            break
        elif key >= 48 and key <= 57: # digit
            digit = key - 48
            number = number * 10 + digit
        elif key == 13:
            break
        elif key == 8:
            return False
        elif key == 113: # 'q'
            save_annotations()
            exit(0)
        else:
            print("Invalid key: %d" % key)

    print("Number: " + str(number))
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
    else:
        index = index - 1
        count = count - 1

save_annotations()
