import tensorflow as tf

from utils import read_examples_list

config_annotations_dir = "data_annotated"
config_train_dir = "data_train"

def process_image(image_path, annotations):
    new_annotations = []
    # TODO: Resize and update annotations
    print(image_path)
    for annotation in annotations:
        print("  %s" % annotation)

def write_csv_file(examples):
    # TODO: Save examples in CSV format
    return

def process_all_examples(examples):
    all_image_paths = list(set(map(lambda e: e["image_path"], examples)))
    resized_examples = []

    for image_path in all_image_paths:
        annotations = []
        for example in examples:
            if example["image_path"] == image_path:
                annotations.append(example)
        resized_examples.extend(process_image(image_path, annotations))

    write_csv_files(resized_examples)

def main(_):
    examples_list = read_examples_list(config_annotations_dir)

    process_all_examples(examples_list)
    # for example in examples_list:
    #     print(example)

if __name__ == '__main__':
    tf.app.run()
