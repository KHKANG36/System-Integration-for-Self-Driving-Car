from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys

import tensorflow as tf


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.TrafficLight=None 


        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        PATH_TO_CKPT = 'model/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = 'model/traffic_label.pbtxt'

        NUM_CLASSES = 3

        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        def load_image_into_numpy_array(image):

               (im_width, im_height) = image.size
               return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            #currently the image is read by feeding the path of the image directory
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                                   [detection_boxes, detection_scores, detection_classes, num_detections],
                                   feed_dict={image_tensor: image_np_expanded})


            _,class_name=vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)
        if class_name == 'Green':
                  return TrafficLight.GREEN
        elif class_name == 'Red':
                   return TrafficLight.RED
        elif class_name == 'Yellow':
                   return TrafficLight.YELLOW
        else:
           return TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLClassifier()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
