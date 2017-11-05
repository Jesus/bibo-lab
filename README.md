We're trying to train a bib detector here.

## Directory structure

  - `data_raw` is just the place where the images are downloaded and wait to be
    annotated.
  - `data_train` has a collection of images that have annotations in the format
    given by VGG Image Annotator. These images are ready to be processed by
    `create_tf_records.py` to build the TF examples.
  - `data_bibs` has a collection of bibs, these are outputs of `detect_bib.py`.
  - `model` contains the output of `create_tf_records.py`, these are examples
    ready to be used by TF for training.

## Train collection structure (`data_train`)

The images are grouped in directories by event. Each directory contains a few
images and one `.csv` file which contains the annotations.

This structure needs to be kept strictly or the `create_tf_records.py` script
won't work correctly.

## How to start training

 1. If required on your environment, activate Tensor Flow:
    `source tensorflow/bin/activate`.
 2. Set up the model: `bash run/setup_model.sh`.
 3. Start the training job: `bash run/train.sh`.
 4. Start the evaluation job: `bash run/eval.sh`.
