import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))
import io
import math
import json
from PIL import Image
from PIL import ImageDraw

from utils import read_examples_list
from config import bibs_path

annotations_path = "data/bib_detection/annotations"
ground_truth_bibs_path = os.path.join(config.bibs_path, "ground_truth")

def process_image(image_path, annotations):
    new_annotations = []
    count = 0

    for annotation in annotations:
        image = Image.open(image_path)

        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')

        new_image_path = image_path.replace(annotations_path, ground_truth_bibs_path)
        new_image_path = new_image_path.replace(".jpg", "-%02d.jpg" % count)
        new_image_path = new_image_path.replace(".JPG", "-%02d.JPG" % count)
        count = count + 1

        box        = json.loads(annotation["raw"][5])
        extra_args = json.loads(annotation["raw"][6])

        new_annotation = {}
        x = new_annotation["x"] = math.floor(box["x"])
        y = new_annotation["y"] = math.floor(box["y"])
        w = new_annotation["width"] = math.floor(box["width"])
        h = new_annotation["height"] = math.floor(box["height"])

        new_annotations.append(new_annotation)

        # Crop on the picture
        image = image.crop((x, y, x + w, y + h))

        new_image_dir = os.path.dirname(new_image_path)
        if not os.path.isdir(new_image_dir):
            os.makedirs(new_image_dir)

        print(new_image_path)
        image.save(new_image_path)

    return new_annotations

def process_all_examples(examples):
    all_image_paths = list(set(map(lambda e: e["image_path"], examples)))
    processed_examples = []

    for image_path in all_image_paths:
        annotations = []
        for example in examples:
            if example["image_path"] == image_path:
                annotations.append(example)

        processed_examples.extend(process_image(image_path, annotations))


if __name__ == '__main__':
    examples_list = read_examples_list(annotations_path)

    process_all_examples(examples_list)
