We're trying to train a bib detector here.

## Directory structure

  - `data_raw` is just the place where the images are downloaded and wait to be
    annotated.
  - `data` has annotated data.
     - `data/bib_detection/annotations` has a collection of images in the format
        given by VGG Image Annotator. These images are ready to be processed by
        `create_tf_records.py` to build the TF examples.
      - `data/number_recognition/annotations` contains images of bibs. Some
        of the images were cropped using the data from ground truth for bib
        detection and others were cropped using the predictions of the bib
        detector. All annotations are kept in a single file `bibs.csv`.


## Training the bib detector

### Annotations

The images are grouped in directories by event. Each directory contains a few
images and one `.csv` file which contains the annotations.

This structure needs to be kept strictly or the `create_tf_records.py` script
won't work correctly.

### How to start training

 1. If required on your environment, activate Tensor Flow:
    `source tensorflow/bin/activate`.
 2. Set up the model: `bash run/setup_model.sh`.
 3. Start the training job: `bash run/train.sh`.
 4. Start the evaluation job: `bash run/eval.sh`.

## Training the number recognizer

TODO
