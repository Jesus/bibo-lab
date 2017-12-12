import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))
import config

import csv
import random
import time
import cv2

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def encode_digit(digit):
    charset = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9
    }
    return charset[digit]

def encode_number(number):
    null_char_code = 10
    length = 6

    char_ids_padded = [null_char_code] * length
    char_ids_unpadded = [null_char_code] * len(number)

    for index in range(length):
        if index < len(number):
            char_code = encode_digit(number[index])
            char_ids_padded[index] = char_ids_unpadded[index] = char_code
        else:
            char_ids_padded[index] = null_char_code

    return char_ids_padded, char_ids_unpadded

def create_tf_example(image_path, number):
    # Implemented thanks to this SO answer by Alex Gorban:
    # https://stackoverflow.com/a/44461910/814224
    image = cv2.imread(image_path)
    image = cv2.resize(image, (80, 80))
    _, jpeg_image = cv2.imencode('.jpeg', image)

    char_ids_padded, char_ids_unpadded = encode_number(number)

    return tf.train.Example(features=tf.train.Features(
      feature={
        'image/format': _bytes_feature("PNG".encode('utf8')),
        'image/encoded': _bytes_feature(jpeg_image.tostring()),
        'image/class': _int64_list_feature(char_ids_padded),
        'image/unpadded_class': _int64_list_feature(char_ids_unpadded),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'orig_width': _int64_feature(image.shape[1]),
        'image/text': _bytes_feature(number.encode('utf8'))
      }
    ))

def create_tf_record(output_filename, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)

    for image_path, number in examples.items():
        tf_example = create_tf_example(image_path, number)
        writer.write(tf_example.SerializeToString())

    writer.close()

def get_examples_split():
    test_p = 0.06
    validation_p = 0.06

    example_paths = []
    examples = {}
    with open(config.annotations_path, "r") as csv_file:
        annotation_reader = csv.reader(csv_file, delimiter = ",")
        for annotation in annotation_reader:
            example_path, _, number = annotation

            if number == "":
                continue

            if example_path in example_paths:
                print(f"ERROR: Duplicate path '{example_path}'")
                exit(1)

            example_paths.append(example_path)
            examples[example_path] = number

    test_c       = int(len(example_paths) * test_p)
    validation_c = int(len(example_paths) * validation_p)
    train_c      = len(example_paths) - test_c - validation_c

    print("Train      : %i" % train_c)
    print("Test       : %i" % test_c)
    print("Validation : %i" % validation_c)
    print("Total      : %i" % len(example_paths))

    random.seed(int(round(time.time() * 1000)))
    random.shuffle(example_paths)

    train_examples = {}
    test_examples = {}
    validation_examples = {}
    for index in range(len(examples)):
        example_path = example_paths[index]
        number = examples[example_path]

        if index < train_c:
            train_examples[example_path] = number
        elif index < (train_c + test_c):
            test_examples[example_path] = number
        else:
            validation_examples[example_path] = number

    return (train_examples, test_examples, validation_examples)

def main(_):
    train_examples, test_examples, validation_examples = get_examples_split()

    # Training set
    create_tf_record(os.path.join(config.model_path, 'train.record'),
        train_examples)

    # Test set
    create_tf_record(os.path.join(config.model_path, 'test.record'),
        test_examples)

    # Validation set
    create_tf_record(os.path.join(config.model_path, 'validation.record'),
        validation_examples)

if __name__ == '__main__':
    tf.app.run()
