import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))
import config

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.platform import flags

import common_flags
import datasets
import model as attention_ocr

FLAGS = flags.FLAGS
common_flags.define()

file_pattern = "models/attention_ocr/python/datasets/data/bibo/tmp/bib-%02d.jpg"

def load_images():
    images_actual_data = np.ndarray(
            shape=(config.batch_size, config.height, config.width, 3),
            dtype='float32')
    for i in range(8):
        path = file_pattern % i
        print("Reading %s" % path)
        image = cv2.imread(path)
        image = cv2.resize(image, (config.width, config.height))
        image = image / 255.0

        images_actual_data[i, ...] = image

    return images_actual_data

def load_graph():
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(config.frozen_inference_graph_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph

def main(_):
    graph = load_graph()
    images_data = load_images()

    images_placeholder = graph.get_tensor_by_name('images_placeholder:0')
    chars_logits_tensor = graph.get_tensor_by_name('output/chars_logits:0')
    chars_tensor = graph.get_tensor_by_name('AttentionOcr_v1/predicted_chars:0')

    with tf.Session(graph=graph) as sess:
        chars_logits, chars = sess.run(
                [chars_logits_tensor, chars_tensor],
                feed_dict={images_placeholder: images_data})

    for text in chars:
        print(text)

if __name__ == '__main__':
    tf.app.run()
