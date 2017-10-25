source run/config.sh

CHECKPOINT_NUMBER=106228

python3 "$OBJECT_DETECTION_PATH/export_inference_graph.py" \
  --input_type="image_tensor" \
  --pipeline_config_path="config/bib_detector.config" \
  --trained_checkpoint_prefix="model/train/model.ckpt-$CHECKPOINT_NUMBER" \
  --output_directory="model/bibo_frozen_inference_graph.pb"
