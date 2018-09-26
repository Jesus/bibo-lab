OBJECT_DETECTION_PATH="models/object_detection"
CHECKPOINT_NUMBER=106228

python3 "$OBJECT_DETECTION_PATH/export_inference_graph.py" \
  --input_type="image_tensor" \
  --input_shape="-1,-1,-1,3" \
  --pipeline_config_path="config/bib_detector.config" \
  --trained_checkpoint_prefix="data/bib_detection/tf_model/train/model.ckpt-$CHECKPOINT_NUMBER" \
  --output_directory="data/bib_detection/tf_model/export"
