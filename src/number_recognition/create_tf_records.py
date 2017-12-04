import tensorflow as tf

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def encode_number_string(number):
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
    null_char_id = 10
    length = 6

    # TODO...
    #                    a b c - -
    # char_ids_padded = [0,1,2,3,3]
    # char_ids_unpadded = [0,1,2]
    char_ids_padded = []
    char_ids_unpadded = []
    return char_ids_padded, char_ids_unpadded

def create_tf_example(image_path, number):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (80, 80))

    number = str(number)
    char_ids_padded, char_ids_unpadded = encode_number_string(number)

    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image/format': _bytes_feature("png"),
        'image/encoded': _bytes_feature(image.tostring()),
        'image/class': _int64_feature(char_ids_padded),
        'image/unpadded_class': _int64_feature(char_ids_unpadded),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'orig_width': _int64_feature(image.shape[1]),
        'image/text': _bytes_feature(number)
      }
    ))

def create_tf_record(output_filename, examples):
    writer = tf.python_io.TFRecordWriter(output_filename)

    for example in examples:
        image_path, number = example
        tf_example = create_tf_example(image_path, number)
        writer.write(tf_example.SerializeToString())

    writer.close()

def split_examples():
    return ([], [], [])

def main(_):
    train_examples, test_examples, validation_examples = get_examples()

    # Training set
    create_tf_record(os.path.join(config_output_dir, 'train.record'),
        train_examples)

    # Test set
    create_tf_record(os.path.join(config_output_dir, 'test.record'),
        test_examples)

    # Validation set
    create_tf_record(os.path.join(config_output_dir, 'validation.record'),
        label_map_dict,
        validation_examples)

if __name__ == '__main__':
    tf.app.run()
