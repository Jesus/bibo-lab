import io
import os
import math
import json
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf

from utils import read_examples_list

config_annotations_dir = "data_annotated"
config_train_dir = "data_train"

MAX_LENGTH = 900
DRAW = False

def process_image(image_path, annotations):
    image = Image.open(image_path)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width, height = image.size
    if width > MAX_LENGTH and width > height:
        resize_scale = float(MAX_LENGTH) / width
    elif height > MAX_LENGTH:
        resize_scale = float(MAX_LENGTH) / height
    else:
        resize_scale = 1.0

    new_width  = math.floor(width * resize_scale)
    new_height = math.floor(height * resize_scale)

    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    new_image_path = image_path.replace(config_annotations_dir, config_train_dir)
    if DRAW:
        draw = ImageDraw.Draw(image)

    new_annotations = []

    # print("%-80s (%ix%i) -> (%ix%i)" % (image_path, width, height, new_width, new_height))
    for annotation in annotations:
        box        = json.loads(annotation["raw"][5])
        extra_args = json.loads(annotation["raw"][6])

        new_annotation = {}
        new_annotation["image_path"] = new_image_path.replace(config_train_dir + "/", "")
        x = new_annotation["x"] = math.floor(box["x"] * resize_scale)
        y = new_annotation["y"] = math.floor(box["y"] * resize_scale)
        w = new_annotation["width"] = math.floor(box["width"] * resize_scale)
        h = new_annotation["height"] = math.floor(box["height"] * resize_scale)
        new_annotation["difficult"] = "difficult" in extra_args and extra_args["difficult"] == 'y'

        new_annotations.append(new_annotation)

        # Draw on the picture
        if DRAW:
            top_left     = (x, y)
            top_right    = (x + w, y)
            bottom_right = (x + w, y + h)
            bottom_left  = (x, y + h)
            if new_annotation["difficult"]:
                color = "red"
            else:
                color = "green"
            draw.polygon([top_left, top_right, bottom_right, bottom_left], outline=color)

    new_image_dir = os.path.dirname(new_image_path)
    if not os.path.isdir(new_image_dir):
        os.makedirs(new_image_dir)

    image.save(new_image_path)

    return new_annotations

def write_csv_file(examples):
    print("Annotations        : %i" % len(examples))

    with open("data_train/annotations.csv", "w") as f:
        # f.writelines(["# This file was generated by `create_train_data.py`"])
        for example in examples:
            if example["difficult"]:
                difficult = "y"
            else:
                difficult = "n"
            f.writelines(['%s,0,\"{}\",1,0,"{""name"":""rect"",""x"":%i,""y"":%i,""width"":%i,""height"":%i}","{""difficult"":""%s""}"\n' % (
                example["image_path"], # file name
                example["x"],       # region shape attributes (x)
                example["y"],       # region shape attributes (y)
                example["width"],   # region shape attributes (width)
                example["height"],  # region shape attributes (height)
                difficult           # region attributes (difficult)
                )])

def process_all_examples(examples):
    all_image_paths = list(set(map(lambda e: e["image_path"], examples)))
    resized_examples = []

    print("Training files     : %i" % len(all_image_paths))
    for image_path in all_image_paths:
        annotations = []
        for example in examples:
            if example["image_path"] == image_path:
                annotations.append(example)

        resized_examples.extend(process_image(image_path, annotations))

    write_csv_file(resized_examples)

def main(_):
    examples_list = read_examples_list(config_annotations_dir)

    process_all_examples(examples_list)

if __name__ == '__main__':
    tf.app.run()
