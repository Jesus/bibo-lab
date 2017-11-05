from PIL import Image
from glob import glob
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

PATH_TO_CKPT = 'model/bibo_frozen_inference_graph.pb'
PATH_TO_LABELS = 'model/label_map.pbtxt'

class BibDetector(object):
    def __init__(self):
        self.detection_graph = self._build_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _build_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def load_image_as_numpy_array(self, image):
        width, height = image.size

        np_array = np.array(image.getdata())
        np_array = np_array.reshape((height, width, 3))
        np_array = np_array.astype(np.uint8)

        return np_array

    def detect_bibs(self, input_path):
        print(f"processing image: {input_path}")
        # Definite input and output Tensors for detection_graph
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image = Image.open(input_path)
        w, h = image.size
        image_np = self.load_image_as_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        print("start sess")
        (boxes, scores, classes, num) = self.sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
        print("sess completed")

        boxes, scores, classes, num = map(np.squeeze, [boxes, scores, classes, num])

        bibs = []
        for i in range(int(num)):
            if scores[i] < 0.001: continue
            score = int(scores[i] * 1000)

            print("")
            print(f"box: {boxes[i]}")
            print(f"score: {scores[i]}")
            print(f"class: {self.category_index[classes[i]]['name']}")

            ymin, xmin, ymax, xmax = boxes[i]

            top     = ymin * h
            left    = xmin * w
            right   = xmax * w
            bottom  = ymax * h
            if right - left < 1: continue
            if bottom - top < 1: continue

            print(f"top: {top}")
            print(f"left: {left}")
            print(f"right: {right}")
            print(f"bottom: {bottom}")
            print(f"width: {w}")
            print(f"height: {h}")

            bib = image.crop((left, top, right, bottom))
            bibs.append(bib)

        return bibs

originals_path = "data/char_analysis/originals"
bibs_path      = "data/char_analysis/bibs"

detector = BibDetector()

for image_path in glob(os.path.join(originals_path, "**/*.jpg"), recursive=True):
    output_file_path = image_path.replace(originals_path, bibs_path)
    output_file_path = os.path.dirname(output_file_path)
    output_file_name = os.path.basename(image_path)

    if not os.path.isdir(output_file_path):
        os.makedirs(output_file_path)

    if os.path.isfile(f"{output_file_path}/{output_file_name}".replace(".jpg", "-00.jpg")):
        print(f"Already processed {image_path}")
        continue

    bibs = detector.detect_bibs(image_path)
    for idx, bib in enumerate(bibs):
        output_path = f"{output_file_path}/{output_file_name}"
        output_path = output_path.replace(".jpg", ("-%02d.jpg" % idx))
        bib.save(output_path)
