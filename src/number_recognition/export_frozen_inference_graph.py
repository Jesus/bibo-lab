import os, sys
sys.path.insert(0, os.path.normpath(os.path.dirname(__file__) + "/.."))
sys.path.insert(0, 'models/attention_ocr/python')
import config

import tensorflow as tf
from tensorflow.python.platform import flags

import common_flags
import model as attention_ocr

FLAGS = flags.FLAGS
common_flags.define()

def load_model(checkpoint, batch_size, dataset_name):
    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    model = common_flags.create_model(
            num_char_classes=dataset.num_char_classes,
            seq_length=dataset.max_sequence_length,
            num_views=dataset.num_of_views,
            null_code=dataset.null_code,
            charset=dataset.charset)
    images_placeholder = tf.placeholder(tf.float32,
            shape=[config.batch_size, config.height, config.width, 3],
            name="images_placeholder")
    endpoints = model.create_base(images_placeholder, labels_one_hot=None)
    init_fn = model.create_init_fn_to_restore(checkpoint)

    return images_placeholder, endpoints, init_fn

def main(_):
    images_placeholder, endpoints, init_fn = load_model(
            config.checkpoint,
            config.batch_size,
            FLAGS.dataset_name)
    with tf.Session() as sess:
        tf.tables_initializer().run()  # required by the CharsetMapper
        init_fn(sess)

        # Available endpoints are: endpoints.chars_logit, endpoints.chars_log_prob,
        # endpoints.predicted_text.
        tf.identity(endpoints.chars_logit, name="output/chars_logits")
        output_nodes = ["output/chars_logits", "AttentionOcr_v1/predicted_chars"]

        # A good part of the code below deserves credit to Morgan Giraud:
        # https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_nodes)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(config.frozen_inference_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    tf.app.run()
