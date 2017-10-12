source run/config.sh

# You can find all available pre-trained models in the `tensorflow/models` repo,
# in `object_detection/g3doc/detection_model_zoo.md`.
MODEL_URL="http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz"
MODEL_PATH="tmp/pre-trained-model.tar.gz"

if [ ! -f "$MODEL_PATH" ]; then
  wget "$MODEL_URL" -O "$MODEL_PATH"
fi

tar zxvf "$MODEL_PATH" -C "model/train" --strip-components=1

python3 create_tf_records.py
