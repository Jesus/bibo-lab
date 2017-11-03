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
    "segmentation/bib-d-1.jpg",
    "segmentation/bib-e-0.jpg",
    "segmentation/bib-e-1.jpg",
    "segmentation/bib-e-10.jpg",
    "segmentation/bib-e-11.jpg",
    "segmentation/bib-e-12.jpg",
    "segmentation/bib-e-13.jpg",
    "segmentation/bib-e-14.jpg",
    "segmentation/bib-e-15.jpg",
    "segmentation/bib-e-16.jpg",
    "segmentation/bib-e-17.jpg",
    "segmentation/bib-e-18.jpg",
    "segmentation/bib-e-19.jpg",
    "segmentation/bib-e-2.jpg",
    "segmentation/bib-e-20.jpg",
    "segmentation/bib-e-21.jpg",
    "segmentation/bib-e-22.jpg",
    "segmentation/bib-e-23.jpg",
    "segmentation/bib-e-24.jpg",
    "segmentation/bib-e-25.jpg",
    "segmentation/bib-e-3.jpg",
    "segmentation/bib-e-4.jpg",
    "segmentation/bib-e-5.jpg",
    "segmentation/bib-e-6.jpg",
    "segmentation/bib-e-7.jpg",
    "segmentation/bib-e-8.jpg",
    "segmentation/bib-e-9.jpg",
    "segmentation/bib-f-0.jpg",
    "segmentation/bib-f-1.jpg",
    "segmentation/bib-f-2.jpg",
    "segmentation/bib-f-3.jpg",
    "segmentation/bib-f-4.jpg",
    "segmentation/bib-f-5.jpg",
    "segmentation/bib-g-0.jpg",
    "segmentation/bib-g-1.jpg",
    "segmentation/bib-g-10.jpg",
    "segmentation/bib-g-11.jpg",
    "segmentation/bib-g-12.jpg",
    "segmentation/bib-g-13.jpg",
    "segmentation/bib-g-14.jpg",
    "segmentation/bib-g-15.jpg",
    "segmentation/bib-g-16.jpg",
    "segmentation/bib-g-2.jpg",
    "segmentation/bib-g-3.jpg",
    "segmentation/bib-g-4.jpg",
    "segmentation/bib-g-5.jpg",
    "segmentation/bib-g-6.jpg",
    "segmentation/bib-g-7.jpg",
    "segmentation/bib-g-8.jpg",
    "segmentation/bib-g-9.jpg")

for image_path in images:
    # load the example image and convert it to grayscale
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=200)
    image = cv2.bilateralFilter(image, 9, 75, 75)
    image_width, image_height, _ = image.shape

    characters = []

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

            width_condition  =     w > 10   and     w < 50
            height_condition =     h > 20   and     h < 100
            ratio_condition  = ratio > 0.2  and ratio < 1

            center   = (x + int(w / 2), y + int(h / 2))
            diagonal = int((w ** 2 + h ** 2) ** 0.5)

            # check to see if the component passes all the tests
            if width_condition and height_condition and ratio_condition:
                characters.append((center, diagonal))

    characters = sorted(characters, key=lambda char: char[0][0])
    if (len(characters) == 0): continue

    # Dimensions of the output character
    output_width  = 50
    output_height = 90
    ratio = output_width / output_height

    # How much context to include around the character
    context = 1.15

    alpha = np.arctan(1 / ratio)
    digits = []
    digits_b = []
    for character in characters:
        ((x, y), d) = character

        d = d * context
        w = d * np.cos(alpha)
        h = d * np.sin(alpha)

        x = x - w / float(2)
        y = y - h / float(2)

        # Move the frame if it overflows the image
        if x < 0: x = 0
        if y < 0: y = 0
        if (x + w) > image_width: x = image_width - w
        if (y + h) > image_height: y = image_height - h

        digit = image[int(y): int(y + h), int(x): int(x + w)]
        digit = cv2.resize(digit, (output_width, output_height))
        digits.append(digit)

        digit_b = binary_adaptive[int(y): int(y + h), int(x): int(x + w)]
        digit_b = cv2.resize(digit_b, (output_width, output_height))
        digits_b.append(digit_b)

    digits   = np.hstack(digits)
    digits_b = np.hstack(digits_b)
    digits_b = np.dstack([digits_b] * 3)

    cv2.imshow("Digits", np.vstack([digits, digits_b]))
    cv2.imshow("Bib", image)
    cv2.waitKey(0)
