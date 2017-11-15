from skimage.filters import threshold_local
from skimage import measure
from glob import glob
import numpy as np
import imutils
import csv
import cv2
import os

bibs_path      = "data/char_analysis/bibs"
chars_path     = "data/char_analysis/chars"

annotations    = "data/char_analysis/annotations.csv"

annotated_files = []
with open(annotations, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")
    for annotation in csv_reader:
        annotated_files.append(annotation[0])

for image_path in glob(os.path.join(chars_path, "**/*.*.jpg"), recursive=True):
    if image_path in annotated_files:
        print(f"Already annotated '{image_path}'")
        continue

    print(image_path)
    file_name = os.path.basename(image_path)
    bib_file_name, coords, bib_file_extension = file_name.split(".")
    x, y, d = coords.split("_")
    x = int(x)
    y = int(y)
    d = int(d)

    region_width  = 50
    region_height = 90
    ratio = region_width / region_height

    alpha = np.arctan(1 / ratio)
    w = int(d * np.cos(alpha))
    h = int(d * np.sin(alpha))

    bib_path = os.path.dirname(image_path)
    bib_path = f"{bib_path}/{bib_file_name}.{bib_file_extension}"

    digit = cv2.imread(image_path)
    bib   = cv2.imread(bib_path)

    bib = cv2.rectangle(bib, (x, y), (x + w, y + h), (0,255,0), 3)

    cv2.imshow("Digit", digit)
    cv2.imshow("Bib", bib)
    key = cv2.waitKey(0)

    if key == 46 or key == 43 or key == 45: # chars '.', '+' & '-'
        digit = None
    elif key >= 48 and key <= 57: # digit
        digit = key - 48
    else:
        exit(1)

    print(f"Digit: {digit}")

    with open(annotations, 'a') as csv_file:
        if digit == None:
            csv_file.write(f"{image_path},n,\n")
        else:
            csv_file.write(f"{image_path},y,{digit}\n")
