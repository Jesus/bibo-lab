"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
import numpy as np
import PIL.Image
import cv2

import tensorflow as tf
from tensorflow.python.platform import flags

import common_flags
import datasets
import model as attention_ocr

FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')


def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
  return width, height

def load_images(file_pattern, batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
                                  dtype='float32')
  for i in range(batch_size):
    path = file_pattern % i
    print("Reading %s" % path)
    image = cv2.imread(path)
    image = cv2.resize(image, (width, height))
    image = image / 255.0
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    images_actual_data[i, ...] = image
  return images_actual_data


def load_model(checkpoint, batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(
      num_char_classes=dataset.num_char_classes,
      seq_length=dataset.max_sequence_length,
      num_views=dataset.num_of_views,
      null_code=dataset.null_code,
      charset=dataset.charset)
  images_placeholder = tf.placeholder(tf.float32,
                                      shape=[batch_size, height, width, 3])
  endpoints = model.create_base(images_placeholder, labels_one_hot=None)
  init_fn = model.create_init_fn_to_restore(checkpoint)
  return images_placeholder, endpoints, init_fn

def main(_):
  images_placeholder, endpoints, init_fn = load_model(FLAGS.checkpoint,
                                                      FLAGS.batch_size,
                                                      FLAGS.dataset_name)
  images_data = load_images(FLAGS.image_path_pattern, FLAGS.batch_size,
                            FLAGS.dataset_name)
  with tf.Session() as sess:
    tf.tables_initializer().run()  # required by the CharsetMapper
    init_fn(sess)
    predictions = sess.run(endpoints.predicted_text,
                           feed_dict={images_placeholder: images_data})
  print("Predicted strings:")
  for idx, prediction in enumerate(predictions):
    image_path = FLAGS.image_path_pattern % idx
    number = prediction.decode('utf8')
    number = number.replace("░", "")

    image = cv2.imread(image_path)
    image = cv2.resize(image, (120, 120))

    footer = np.zeros((40, 120, 3), np.uint8)
    cv2.putText(footer, number, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    image = np.vstack([image, footer])

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
  tf.app.run()
