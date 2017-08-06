# This code is based on the sample one from from TF models repo, in file:
# object_detection/create_pet_tf_record.py

r"""Convert the training data generated using VGG Image Annotator to TFRecord
"""

import logging
import os
import io
import random
from glob import glob
import csv

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

config_train_dir = "data_train"
config_label_map_path = "data_train/label_map.pbtxt"

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

    for example in examples_list:
        print(example["image_path"])

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    # random.seed(42)
    # random.shuffle(examples_list)
    # num_examples = len(examples_list)
    # num_train = int(0.7 * num_examples)
    # train_examples = examples_list[:num_train]
    # val_examples = examples_list[num_train:]
    # logging.info('%d training and %d validation examples.',
    #                            len(train_examples), len(val_examples))
    #
    # train_output_path = os.path.join(FLAGS.output_dir, 'pet_train.record')
    # val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
    # create_tf_record(train_output_path, label_map_dict, annotations_dir,
    #                                    image_dir, train_examples)
    # create_tf_record(val_output_path, label_map_dict, annotations_dir,
    #                                    image_dir, val_examples)

if __name__ == '__main__':
    tf.app.run()
