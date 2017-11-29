from skimage.filters import threshold_local
from skimage import measure
from glob import glob
import numpy as np
import imutils
import cv2
import os

bibs_path      = "data/char_analysis/bibs"
chars_path     = "data/char_analysis/chars"

for image_path in glob(os.path.join(bibs_path, "**/*.jpg"), recursive=True):
    # load the example image and convert it to grayscale
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=200)
    image_orig = image
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
    digits_c = []
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

        # Digit image
        digit = image[int(y): int(y + h), int(x): int(x + w)]
        digit = cv2.resize(digit, (output_width, output_height))
        digits.append(digit)

        # Digit, binary
        digit_b = binary_adaptive[int(y): int(y + h), int(x): int(x + w)]
        digit_b = cv2.resize(digit_b, (output_width, output_height))
        digits_b.append(digit_b)

        # Digit coordinates
        digit_c = ((x, y), d)
        digits_c.append(digit_c)

    # Prepare visualization
    vis_digits   = np.hstack(digits)
    vis_digits_b = np.hstack(digits_b)
    vis_digits_b = np.dstack([vis_digits_b] * 3)
    vis          = np.vstack([vis_digits, vis_digits_b])

    print(f"{image_path}")
    # cv2.imshow("Digits", vis)
    # cv2.imshow("Bib", image)
    # cv2.waitKey(0)

    output_folder_path = image_path.replace(bibs_path, chars_path)
    output_folder_path = os.path.dirname(output_folder_path)

    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)

    file_name   = os.path.basename(image_path)
    output_path = f"{output_folder_path}/{file_name}"
    cv2.imwrite(output_path, image_orig)
    for idx, digit in enumerate(digits):
        ((x, y), d) = digits_c[idx]

        digit_path = output_path.replace(".jpg", (".%d_%d_%d.jpg" % (x, y, d)))

        print(digit_path)
        cv2.imwrite(digit_path, digit)

    print("")
