from PIL import Image
from PIL import ImageDraw
import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

PATH_TO_CKPT = 'model/bibo_frozen_inference_graph.pb'
PATH_TO_LABELS = 'model/label_map.pbtxt'

input_path = "input.jpg"
output_path = "output.jpg"

class ObjectDetector(object):
    def __init__(self):
        self.detection_graph = self._build_graph()
        self.sess = tf.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
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

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect(self, image):
        image_np = self._load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num_detections = map(np.squeeze, [boxes, scores, classes, num_detections])

        return boxes, scores, classes.astype(int), num_detections

detector = ObjectDetector()
image = Image.open(input_path).convert('RGB')
detector.detect(image)
# continue here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# 
# boxes, scores, classes, num_detections =
# for i in range(num_detections):
#     print(boxes[i])
