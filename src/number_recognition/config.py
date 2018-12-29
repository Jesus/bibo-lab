import os

bibs_path        = "data/number_recognition/annotations/bibs"
annotations_path = "data/number_recognition/annotations/bibs.csv"

model_path       = "models/attention_ocr/python/datasets/data/bibo"
train_path       = "models/attention_ocr/python/workdir/train"

checkpoint = os.path.join(train_path, "model.ckpt-93215")
batch_size = 32
width, height = 120, 120

frozen_inference_graph_path = "number_recognition_frozen_inference_graph.pb"
