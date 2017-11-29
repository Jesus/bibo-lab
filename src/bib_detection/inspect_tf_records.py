import os
import tensorflow as tf

config_label_map_path = "model/label_map.pbtxt"
config_train_dir = "data_train"
config_output_dir = "model"

for example in tf.python_io.tf_record_iterator(os.path.join(config_output_dir, 'bibo_train.record')):
    print(tf.train.Example.FromString(example))
    print("-" * 80)
