# This code is based on the sample one from from TF models repo, in file:
# object_detection/create_pet_tf_record.py

r"""Convert the training data generated using VGG Image Annotator to TFRecord
"""

import hashlib
import logging
import os
import io
import random
import time
from glob import glob
import csv
import json

import PIL.Image

import tensorflow as tf

from collections import defaultdict

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from utils import read_examples_list

config_label_map_path = "model/label_map.pbtxt"
config_train_dir = "data_train"
config_output_dir = "model"

def create_tf_example(image_path, examples, label_map_dict):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width, height = image.size
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    difficult_obj = []
    for example in examples:
        # box example:
        # {
        #   "name": "rect",
        #   "x": 294,
        #   "y":549,
        #   "width":57,
        #   "height":60
        # }
        box        = json.loads(example["raw"][5])
        extra_args = json.loads(example["raw"][6])

        _xmin = float(box["x"]) / width
        _ymin = float(box["y"]) / height
        _xmax = float(box["x"] + box["width"]) / width
        _ymax = float(box["y"] + box["height"]) / height
        values = [_xmin, _ymin, _xmax, _ymax]
        for value in values:
            if value < 0 or value >= 1:
                print("Value out of boundaries %s" % example["image_path"])
        xmin.append(_xmin)
        ymin.append(_ymin)
        xmax.append(_xmax)
        ymax.append(_ymax)
        class_name = 'bib'
        classes.append(label_map_dict[class_name])
        classes_text.append(class_name.encode('utf8'))
        if "difficult" in extra_args and extra_args["difficult"] == 'y':
            difficult_obj.append(int(True))
        else:
            difficult_obj.append(int(False))

    return tf.train.Example(features = tf.train.Features(feature = {
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
        }))

def create_tf_record(output_filename,
                     label_map_dict,
                     examples):
    writer = tf.python_io.TFRecordWriter(output_filename)

    # Group examples by their file name
    grouped_examples = {}
    for example in examples:
        if example['image_path'] in grouped_examples:
            grouped_examples[example['image_path']].append(example)
        else:
            grouped_examples[example['image_path']] = [example]

    for path, example_group in grouped_examples.items():
        tf_example = create_tf_example(path, example_group, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()

# Splits the examples in two: Training & validation.
def split_data_set(examples):
    random.seed(int(round(time.time() * 1000)))
    random.shuffle(examples)

    train_examples = []
    eval_examples = []

    temptative_p = 0.7
    all_image_paths = list(set(map(lambda e: e["image_path"], examples)))
    print("Training files     : %i" % len(all_image_paths))
    n_train_image_paths = int(temptative_p * len(all_image_paths))
    train_image_paths = all_image_paths[:n_train_image_paths]
    eval_image_paths = all_image_paths[n_train_image_paths:]

    for example in examples:
        if example["image_path"] in train_image_paths:
            train_examples.append(example)
        elif example["image_path"] in eval_image_paths:
            eval_examples.append(example)
        else:
            raise Exception("Image path not found")

    actual_p = len(train_examples) / float(len(examples)) * 100
    print("Training examples  : %i (%.2f%%)" % (len(train_examples), actual_p))
    print("Validation examples: %i" % len(eval_examples))

    return train_examples, eval_examples

def main(_):
    label_map_dict = label_map_util.get_label_map_dict(config_label_map_path)
    examples_list  = read_examples_list(config_train_dir)

    train_examples, eval_examples = split_data_set(examples_list)

    # Training set
    create_tf_record(os.path.join(config_output_dir, 'bibo_train.record'),
        label_map_dict,
        train_examples)

    # Validation set
    create_tf_record(os.path.join(config_output_dir, 'bibo_eval.record'),
        label_map_dict,
        eval_examples)

if __name__ == '__main__':
    tf.app.run()
