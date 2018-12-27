import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))
import config

import urllib.request
import psycopg2
import csv

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

def create_tf_example(example):
    # {
    #     "image_path": local_path,
    #     "x0": row[3],
    #     "y0": row[4],
    #     "x1": row[5],
    #     "y1": row[6],
    #     "number": row[7]
    # }
    image = cv2.imread(example['image_path'])
    if example['crop']:
        if example['y0'] < example['y1']:
            y_min = example['y0']
            y_max = example['y1']
        else:
            y_min = example['y1']
            y_max = example['y0']

        if example['x0'] < example['x1']:
            x_min = example['x0']
            x_max = example['x1']
        else:
            x_min = example['x1']
            x_max = example['x0']
        image = image[y_min:y_max, x_min:x_max]
    image = cv2.resize(image, (80, 80))
    _, jpeg_image = cv2.imencode('.jpeg', image)

    char_ids_padded, char_ids_unpadded = encode_number(example['number'])

    return tf.train.Example(features=tf.train.Features(
      feature={
        'image/format': _bytes_feature('jpg'.encode('utf8')),
        'image/encoded': _bytes_feature(jpeg_image.tostring()),
        'image/class': _int64_list_feature(char_ids_padded),
        'image/unpadded_class': _int64_list_feature(char_ids_unpadded),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'orig_width': _int64_feature(image.shape[1]),
        'image/text': _bytes_feature(example['number'].encode('utf8'))
      }
    ))

def create_tf_record(output_filename, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()

def get_split_name(file_name):
    # We choose a split (train, test, validation) based on the image file name
    file_name = file_name.lower()
    sum = 0

    for c in file_name:
        sum += ord(c)

    sum = sum % 10

    if sum == 0:
        return "test"
    elif sum == 1:
        return "validation"
    else:
        return "train"

def get_pg_examples():
    if 'DBHOST' in os.environ:
        connection = psycopg2.connect(host=os.environ['DBHOST'],
                                      dbname=os.environ['DBNAME'],
                                      user=os.environ['DBUSER'],
                                      password=os.environ['DBPASS'])
    else:
        connection = psycopg2.connect("dbname=bibo-web-dev")
    cursor = connection.cursor()
    query = """
            SELECT
                photos.id,
                photos.image_file_name,
                bib_annotations.style,
                bib_annotations.x0,
                bib_annotations.y0,
                bib_annotations.x1,
                bib_annotations.y1,
                bib_annotations.number
            FROM bib_annotations, photos
            WHERE bib_annotations.photo_id = photos.id
              AND number IS NOT NULL
              AND number <> ''
              AND photos.bib_numbers_annotated_at IS NOT NULL;
            """

    cursor.execute(query)
    examples = {
        "test": [],
        "validation": [],
        "train": []
    }
    for i, row in enumerate(cursor.fetchall()):
        id              = '%09d' % row[0]
        style           = row[2]
        image_name      = row[1][:-4]
        image_extension = row[1][-4:]

        if image_extension[0] != '.':
            raise RuntimeError("Unexpected image file name format")

        if style != "original":
            image_extension = image_extension.lower()

        url        = f"https://s3-eu-west-1.amazonaws.com/bibo-web/photos/images/{id[0:3]}/{id[3:6]}/{id[6:9]}/{style}/{image_name}{image_extension}"
        local_path = f"tmp/number_recognition/{id}-{style}-{image_name}{image_extension}"
        split_name = get_split_name(f"{image_name}{image_extension}")

        if not os.path.isfile(local_path):
            urllib.request.urlretrieve(url, local_path)

        examples[split_name].append({
            "image_path": local_path,
            "crop": True,
            "x0": int(row[3]),
            "y0": int(row[4]),
            "x1": int(row[5]),
            "y1": int(row[6]),
            "number": row[7]
        })
    cursor.close()

    return examples

def get_csv_examples():
    examples = {
        "test": [],
        "validation": [],
        "train": []
    }
    with open(config.annotations_path, "r") as csv_file:
        annotation_reader = csv.reader(csv_file, delimiter = ",")
        for annotation in annotation_reader:
            example_path, _, number = annotation

            if number == "":
                continue

            split_name = get_split_name(f"{os.path.basename(example_path)}")
            examples[split_name].append({
                "image_path": example_path,
                "crop": False,
                "number": number
            })

    return examples

def show_dataset_stats(examples):
    train_c = len(examples['train'])
    test_c = len(examples['test'])
    validation_c = len(examples['validation'])
    total_c = train_c + test_c + validation_c

    print("Train:       %5i ( %.1f%% )" % (train_c, train_c * 100 / total_c))
    print("Test:        %5i ( %.1f%% )" % (test_c, test_c * 100 / total_c))
    print("Validation:  %5i ( %.1f%% )" % (validation_c, validation_c * 100 / total_c))
    print("")
    print("Total:       %5i" % total_c)

def main():
    pg_examples = get_pg_examples()
    csv_examples = get_csv_examples()
    examples = {
        "test": pg_examples["test"] + csv_examples["test"],
        "validation": pg_examples["validation"] + csv_examples["validation"],
        "train": pg_examples["train"] + csv_examples["train"]
    }
    show_dataset_stats(examples)

    # Training set
    create_tf_record(os.path.join(config.model_path, 'train.record'),
        examples['train'])

    # Test set
    create_tf_record(os.path.join(config.model_path, 'test.record'),
        examples['test'])

    # Validation set
    create_tf_record(os.path.join(config.model_path, 'validation.record'),
        examples['validation'])

if __name__ == '__main__':
    main()
