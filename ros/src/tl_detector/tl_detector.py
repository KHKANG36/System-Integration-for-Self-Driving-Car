#!/usr/bin/env python
#TeamX - Ryan Kang

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:pose (Pose): position to match a waypoint to
        Returns:int: index of the closest waypoint in self.waypoints
        """
        Sudo_Min_Dist = 1e9 
        index = None
        for wp in range(len(self.waypoints)):
            x_dist = pose.position.x - self.waypoints[wp].pose.pose.position.x
            y_dist = pose.position.y - self.waypoints[wp].pose.pose.position.y
            wp_dist = np.sqrt(x_dist**2 + y_dist**2)
            
            if wp_dist < Sudo_Min_Dist:
                Sudo_Min_Dist = wp_dist
                index = wp
        return index
    
    def get_closest_waypoint_stop_line(self, stopline):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:pose (Pose): position to match a waypoint to
        Returns:int: index of the closest waypoint in self.waypoints
        """
        Sudo_Min_Dist = 1e9 
        index = None
        for wp in range(len(self.waypoints)):
            x_dist = stopline[0] - self.waypoints[wp].pose.pose.position.x
            y_dist = stopline[1] - self.waypoints[wp].pose.pose.position.y
            wp_dist = np.sqrt(x_dist**2 + y_dist**2)
            
            if wp_dist < Sudo_Min_Dist:
                Sudo_Min_Dist = wp_dist
                index = wp
        return index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:light (TrafficLight): light to classify

        Returns:int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8") # I changed this from bgr8 to rgb8

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        TL_waypoint_close = []

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        # Find closest waypoint indexs to each stop line (traffic light)
        for i in range(len(stop_line_positions)):
            TL_close = self.get_closest_waypoint_stop_line(self, stop_line_positions[i])
            TL_waypoint_close.append(TL_close)
        
        TL_waypoint_close_sort = TL_waypoint_close
        TL_waypoint_close_sort.sort()
            
        # Find closest next stop line (traffic light) index to me
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose) #My current car position index 
        
        if car_position > max(TL_waypoint_close):
            TL_waypoint_next_me = min(TL_waypoint_close)
        else if car_position < min(TL_waypoint_close):
            TL_waypoint_next_me = min(TL_waypoint_close)
        else:
            for j in range(len(TL_waypoint_close_sort)):
                if (car_position - TL_waypoint_close_sort[j]) > 0:
                    pass
                else:
                    TL_waypoint_next_me = TL_waypoint_close_sort[j]
        
        # Find closest next stop line (traffic light) position to me 
        stop_line_index = TL_waypoint_close.index(TL_waypoint_next_me)
        light = stop_line_position[stop_line_index]
        
        TL_distance = np.sqrt((light[0] - self.waypoints[car_position].pose.pose.position.x)**2 + (light[1] - self.waypoints[car_position].pose.pose.position.y)**2)
        
        # We have to difine the visibility..(with distance)

        if light:
            if light_distance >= 50 #How we can calculate this? Maybe we need simulator information..
                return -1, TrafficLight.UNKNOWN

            else:
                state = self.get_light_state(light)
                return light, state    
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
