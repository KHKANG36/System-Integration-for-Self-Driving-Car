from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2 
import scipy.misc
import numpy as np 

class TLClassifier(object):
    def __init__(self):
        # Please modify this code for your best  
        self.sess = tf.Session()
        #graph = load_graph(filename) #Put the file name

        # Access the output node
        self.output = graph.get_tensor_by_name('trafficlight/model_saved/XXX:0')
        self.image_input_pl = tf.placeholder(tf.int8, (None,640,480,3)) 
        
        pass

    def load_graph(filename): 
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph: 
            tf.import_graph_def(graph_def, name="trafficlight")
        return graph

    def get_classification(self, image):
        """
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #Image resize (if not required) 
	image_shape = (640, 480)
        image = scipy.misc.imresize(scipy.misc.imread(image), image_shape)

        results = self.sess.run([tf.nn.top_k(tf.nn.softmax(self.output))],
                                {self.image_input_pl: [image]})
        light_state = int(np.array(results[0].indices).flatten()[0])

        #Classify 
        if light_state == 0:
           return TrafficLight.RED
        if light_state == 1:
           return TrafficLight.YELLOW
        if light_state == 2:
           return TrafficLight.GREEN
        else: 
           return TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLClassifier(object)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
