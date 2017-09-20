source run/config.sh

python3 "$OBJECT_DETECTION_PATH/eval.py" \
  --logtostderr \
  --pipeline_config_path=config/bib_detector.config \
  --checkpoint_dir=model/train --eval_dir=model/eval \
  | tee -a "log/eval.log" 2>&1
