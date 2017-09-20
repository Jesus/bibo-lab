source run/config.sh

python3 "$OBJECT_DETECTION_PATH/train.py" \
  --logtostderr \
  --pipeline_config_path=config/bib_detector.config \
  --train_dir=model/train \
  | tee -a "log/train.log" 2>&1
