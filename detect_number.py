from PIL import Image
from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import imutils
import cv2
import os

images = (
    "segmentation/bib-a-0.jpg",
    "segmentation/bib-a-1.jpg",
    "segmentation/bib-a-2.jpg",
    "segmentation/bib-a-3.jpg",
    "segmentation/bib-a-4.jpg",
    "segmentation/bib-a-5.jpg",
    "segmentation/bib-a-6.jpg",
    "segmentation/bib-a-7.jpg",
    "segmentation/bib-a-8.jpg",
    "segmentation/bib-b-0.jpg",
    "segmentation/bib-b-1.jpg",
    "segmentation/bib-b-2.jpg",
    "segmentation/bib-c-0.jpg",
    "segmentation/bib-c-1.jpg",
    "segmentation/bib-c-2.jpg",
    "segmentation/bib-c-3.jpg",
    "segmentation/bib-c-4.jpg",
    "segmentation/bib-c-5.jpg",
    "segmentation/bib-c-6.jpg",
    "segmentation/bib-d-0.jpg",
    "segmentation/bib-d-1.jpg")

for image in images:
    # load the example image and convert it to grayscale
    image = cv2.imread(image)
    image = imutils.resize(image, width=200)
    image = cv2.bilateralFilter(image, 9, 75, 75)

    # extract the Value component from the HSV color space and apply adaptive thresholding
    # to reveal the characters on the license plate
    image_value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    adaptive_thresh = threshold_local(image_value, 45, offset=15)
    binary_adaptive = image_value < adaptive_thresh
    binary_adaptive = binary_adaptive.astype("uint8") * 255

    # binary_adaptive = cv2.bitwise_not(binary_adaptive)

    # perform a connected components analysis and initialize the mask to store
    # the locations of the character candidates
    labels = measure.label(binary_adaptive, neighbors=8, background=0)
    charCandidates = np.zeros(binary_adaptive.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        # if label == -1: continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(binary_adaptive.shape, dtype="uint8")
        labelMask[labels == label] = 255
        _, contours, heirs = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # ensure at least one contour was found in the mask
        if len(contours) > 0:
            # grab the largest contour which corresponds to the component in
            # the mask, then grab the bounding box for the contour
            c = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w / float(h)

            width_condition  =     w > 15   and     w < 50
            height_condition =     h > 30   and     h < 100
            ratio_condition  = ratio > 0.2  and ratio < 1

            # check to see if the component passes all the tests
            if width_condition and height_condition and ratio_condition:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    candidates = np.dstack([charCandidates] * 3)
    threshold = np.dstack([binary_adaptive] * 3)
    output = np.vstack([image, threshold, candidates])
    cv2.imshow("Bib", output)
    cv2.waitKey(0)

    # Continue with:
    #   - https://gurus.pyimagesearch.com/lesson-sample-segmenting-characters-from-license-plates/#

    # Related:
    #   - https://stackoverflow.com/questions/41670628/segmentation-of-lines-words-and-characters-from-a-documents-image
    #   - https://stackoverflow.com/questions/40443988/python-opencv-ocr-image-segmentation
