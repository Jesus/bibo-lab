import os
import csv
from glob import glob

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
    csv_files = glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

    for csv_file in csv_files:
        examples += read_examples_list_from_csv(csv_file)

    return examples
