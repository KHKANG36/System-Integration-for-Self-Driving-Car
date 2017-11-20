from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        light_state=classify(image)
        if light_state == 0:
           return TrafficLight.RED
        if light_state == 1:
           return TrafficLight.YELLOW
        if light_state == 2:
           return TrafficLight.GREEN
        else:
           return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN
