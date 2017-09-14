# This code is based on the sample one from from TF models repo, in file:
# object_detection/create_pet_tf_record.py

r"""Convert the training data generated using VGG Image Annotator to TFRecord
"""

import hashlib
import logging
import os
import io
import random
from glob import glob
import csv
import json

import PIL.Image

import tensorflow as tf

from collections import defaultdict

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

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
    for example in examples:
        # box example:
        # {
        #   "name": "rect",
        #   "x": 294,
        #   "y":549,
        #   "width":57,
        #   "height":60
        # }
        box = json.loads(example["raw"][5])

        xmin.append(float(box["x"]))
        ymin.append(float(box["y"]))
        xmax.append(float(box["x"] + box["width"]))
        ymax.append(float(box["y"] + box["height"]))
        class_name = 'bib'
        classes.append(label_map_dict[class_name])
        classes_text.append(class_name.encode('utf8'))

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


def read_examples_list_from_csv(csv_path):
    dirname = os.path.dirname(csv_path)
    examples = []

    # CSV structure:
    #
    # idx field
    # --- -----------------------------------
    #   0 filename
    #   1 file_size
    #   2 file_attributes
    #   3 region_count
    #   4 region_id
    #   5 region_shape_attributes
    #   6 region_attributes
    with open(csv_path, "r") as csvfile:
        annotation_reader = csv.reader(csvfile, delimiter = ",")
        for annotation in annotation_reader:
            if annotation[0][0] == "#":
                continue # This is a comment line...
            if int(annotation[3]) == 0:
                continue

            examples.append({
                "raw": annotation,
                "image_path": os.path.join(dirname, annotation[0])
            })

    return examples

def read_examples_list(data_dir):
    examples = []

    for csv_file in glob(os.path.join(data_dir, "*/*.csv")):
        examples += read_examples_list_from_csv(csv_file)

    return examples

def main(_):
    label_map_dict = label_map_util.get_label_map_dict(config_label_map_path)

    logging.info('Reading dataset.')
    examples_list = read_examples_list(config_train_dir)

    # Split set in two: Training & validation.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    print("Training examples  : %i" % len(train_examples))
    print("Validation examples: %i" % len(val_examples))

    # Training set
    create_tf_record(os.path.join(config_output_dir, 'bibo_train.tfrecord'),
        label_map_dict,
        train_examples)

    # Validation set
    create_tf_record(os.path.join(config_output_dir, 'bibo_val.tfrecord'),
        label_map_dict,
        val_examples)

if __name__ == '__main__':
    tf.app.run()
