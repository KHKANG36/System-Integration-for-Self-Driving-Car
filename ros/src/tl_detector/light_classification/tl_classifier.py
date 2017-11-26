from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys

import tensorflow as tf


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.TrafficLight=None 
        PATH_TO_CKPT = '/home/student/CarND-Capstone-Master/ros/src/tl_detector/light_classification/model/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = '/home/student/CarND-Capstone-Master/ros/src/tl_detector/light_classification/model/traffic_label.pbtxt'

        NUM_CLASSES = 3

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          #sess=tf.Session(graph=self.detection_graph) 
          with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
          self.sess=tf.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)


        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #print("Detection commencing")
        with self.detection_graph.as_default():
            #currently the image is read by feeding the path of the image directory
            #image_np = load_image_into_numpy_array(image)
            image_np = np.asarray(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = self.sess.run(
                                   [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                   feed_dict={self.image_tensor: image_np_expanded})

            
            _,class_name=vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)
        print (class_name)
        if class_name == 'Green':
                  self.TrafficLight= TrafficLight.GREEN
        elif class_name == 'Red':
                  self.TrafficLight= TrafficLight.RED
        elif class_name == 'Yellow':
                  self.TrafficLight= TrafficLight.YELLOW
        else:
                 self.TrafficLight=TrafficLight.UNKNOWN

        return self.TrafficLight         

if __name__ == '__main__':
    try:
        TLClassifier()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
