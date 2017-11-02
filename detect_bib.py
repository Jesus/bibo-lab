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

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image = Image.open(input_path)
        w, h = image.size
        draw = ImageDraw.Draw(image)
        image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        boxes, scores, classes, num = map(np.squeeze, [boxes, scores, classes, num])

        for i in range(int(num)):
            if scores[i] < 0.001: continue
            score = int(scores[i] * 1000)

            print("")
            print(f"box: {boxes[i]}")
            print(f"score: {scores[i]}")
            print(f"class: {category_index[classes[i]]['name']}")

            ymin, xmin, ymax, xmax = boxes[i]

            # top_left     = (xmin * w, ymin * h)
            # top_right    = (xmax * w, ymin * h)
            # bottom_right = (xmax * w, ymax * h)
            # bottom_left  = (xmin * w, ymax * h)
            # draw.polygon([top_left, top_right, bottom_right, bottom_left], outline="red")
            # image.save(output_path)

            top     = ymin * h
            left    = xmin * w
            right   = xmax * w
            bottom  = ymax * h

            bib = image.crop((left, top, right, bottom))
            bib.save(f"output/bib-{i}-{score}.jpg")
