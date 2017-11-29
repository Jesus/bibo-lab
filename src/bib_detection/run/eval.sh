source run/config.sh


export CUDA_VISIBLE_DEVICES="-1"

python3 "$OBJECT_DETECTION_PATH/eval.py" \
  --logtostderr \
  --pipeline_config_path=config/bib_detector.config \
  --checkpoint_dir=model/train --eval_dir=model/eval \
  2>&1 | tee -a "log/eval.log"
