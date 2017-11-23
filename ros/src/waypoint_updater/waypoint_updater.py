#!/usr/bin/env python
#Team X - Greg McCluskey

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tensorflow as tf
from tf.transformations import euler_from_quaternion

from copy import deepcopy

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
OPTIMAL_BREAKING_DIST = 100  # Normal zone for gradual breaking for the car 
PI = 3.141593
PIOVER2 = 1.570796
PI1POINT5 = 4.712389   #1.5 * pi 

CHECK_AHEAD_NUM_INDEXES = 50
PROXIMITY_THRESH = 25

LAGBUMPER = 0



class WaypointUpdater(object):
    def __init__(self):
        #rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)
        rospy.init_node('waypoint_updater')

        self.base_waypoints = None
        self.lengthBaseWPs = 0

        self.stopLightIdx = -1
        self.obstacleIdx = -1

	self.prevWPIdx = 0
	

        self.oldpos = None

        self.current_pose_sub = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        #self.base_waypoints_sub = None

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # /obstacle?
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        rospy.spin()



    def get_euler(self, inpose):
        quaternion = (inpose.orientation.x, inpose.orientation.y, 
                      inpose.orientation.z, inpose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        #roll, pitch, yaw
        return euler[0], euler[1], euler[2]


    def get_angle_to_point(self, carpose, wppose):
	y = wppose.position.y - carpose.position.y
        x = wppose.position.x - carpose.position.x

	angle = math.atan2(y,x)
        #rospy.logdebug("arctan===%s  y=%s, x=%s", angle, y, x)
      	return angle

    def get_nearest_brute(self, idxStart, idxEnd, curPos):
	lowestDist = 999999.0
        nearestWPIdx = 0  

	#rospy.logdebug("Brute find")

        if idxEnd + 1 >= self.lenBaseWPs:
		idxTheEnd = self.lenBaseWPs -1
  	else: 
		idxTheEnd = idxEnd

	if idxStart >= idxTheEnd:
		return idxStart

        #rospy.logdebug("Brute: startidx=%s, endidx=%s", idxStart, idxTheEnd)
        
        for i in range(idxStart, idxTheEnd+1):
 		d = self.dist(self.base_waypoints[i].pose.pose.position, curPos)
		if( d < lowestDist ):
			lowestDist = d
                        nearestWPIdx = i

        #rospy.logdebug("Brute: Lowest dist found=%s, idx=%s", lowestDist, nearestWPIdx) 
	return nearestWPIdx


    def find_wp_ahead(self, carpose):
        
        # Find distance to last used previous index
        d = self.dist(self.base_waypoints[self.prevWPIdx].pose.pose.position, carpose.position)

	if d < PROXIMITY_THRESH:
		# Use the previous index as the starting point to search for closest point
		nearestWPIdx = self.get_nearest_brute(self.prevWPIdx-1, self.prevWPIdx + CHECK_AHEAD_NUM_INDEXES, carpose.position)
                #bruteIdx = self.get_nearest_brute(0, self.lenBaseWPs-1, carpose.position)
		#rospy.logwarn("Dist=%s  PrevWP=%s  NewWP=%s, bruteWP=%s", d, self.prevWPIdx, nearestWPIdx, bruteIdx)
                

                #double check the found one
                #d = self.dist(self.base_waypoints[nearestWPIdx].pose.pose.position, carpose.position)
		#if d > PROXIMITY_THRESH:
	        #	rospy.logwarn("Double check")
	        #	# Have to check all potential wps
  		#	nearestWPIdx = self.get_nearest_brute(0, self.lenBaseWPs-1, carpose.position)
			
	else:
		# Have to check all potential wps
  		nearestWPIdx = self.get_nearest_brute(0, self.lenBaseWPs-1, carpose.position)        
		rospy.logdebug("Checking all points!!!! dist=%s  prev idx=%s", d, self.prevWPIdx)
	
	#rospy.logdebug("=================================")
	#rospy.logdebug("Nearest idx=%s.", nearestWPIdx)
        #rospy.logdebug("car pos= %s" , carpose)
        #rospy.logdebug("waypoint pos= %s" , base_waypoints[nearestWPIdx].pose.pose)

        # Determine if point in front of car...

        # Convert quaternion orientation to euler for car 
        carroll, carpitch, caryaw = self.get_euler(carpose)
        
        # Get angle from car to closest waypoint
  	angleToWP = self.get_angle_to_point(carpose, self.base_waypoints[nearestWPIdx].pose.pose)

	#rospy.logwarn("Angle to waypoint=%s", angleToWP)
	#rospy.logdebug("Angle to waypoint=%s", angleToWP)

        # Determine if closest waypoint is behind the car.  
        yawDiff = abs(caryaw - angleToWP) 
        if (yawDiff >= PIOVER2) and (yawDiff <= PI1POINT5):
		#Car location is ahead of first waypoint so increment waypoint index assuming next waypoint would be ahead
		#rospy.logdebug("Car is ahead of closest waypoint.  Bumping increment.")
		nearestWPIdx+=1  
		#rospy.logwarn("Bumped")
		

        self.prevWPIdx = nearestWPIdx	

        #rospy.logdebug("Adjusted Nearest idx=%s.", nearestWPIdx)

        return nearestWPIdx


    def getClosestStopIdx(self):
        idx = -1
	if self.stopLightIdx > -1:
      		idx = self.stopLightIdx
          	if self.obstacleIdx > -1:
			idx = min(idx, self.objstacleIdx)
	elif self.obstacleIdx > -1:
		idx = self.obstacleIdx	
        return idx
     

    def pose_cb(self, msg):
        # TODO: Implement
	if( self.base_waypoints is None ):
		# no base waypoints issued yet
		rospy.logwarn("Waiting for base waypoints...")
		return
	
   	#rospy.logdebug("begin msg pos x=%s, y=%s", msg.pose.position.x, msg.pose.position.y) 
	
        #rospy.logwarn("Before")



        #rate = rospy.Rate(50) # 50Hz


        # Do not process get nearest waypoint again if in same position
        if self.oldpos is None:
                # Find nearest waypoint ahead of car
                nearestWPIdx = self.find_wp_ahead(msg.pose)
                # Save position
                self.oldpos = deepcopy(msg.pose)
                #rospy.logdebug("First old pose=%s", self.oldpos)                 
        elif (self.oldpos.position.x == msg.pose.position.x) and (self.oldpos.position.y == msg.pose.position.y):
                nearestWPIdx = self.prevWPIdx
                #rospy.logdebug("Same as old position car pos x=%s", self.oldpos.position.x) 
                #rospy.logwarn("old== returning")
        else:
		# Find nearest waypoint ahead of car
                nearestWPIdx = self.find_wp_ahead(msg.pose)
                # Save position
                self.oldpos = deepcopy(msg.pose)
                #rospy.logdebug("New position of car. pos x=%s", self.oldpos.position.x) 
        

	nearestWPIdx += LAGBUMPER
	
        # Put wanted waypoints in control WP list
        cntrl_waypoints = Lane()

 	endWPIdx = nearestWPIdx+LOOKAHEAD_WPS
	if( endWPIdx >= self.lenBaseWPs):  #dont go beyond waypoint list
		endWPIdx = self.lenBaseWPs-1

        stopWPIdx = self.getClosestStopIdx()
        #rospy.logwarn("stop idx...%s", stopWPIdx)
        if( stopWPIdx != -1 and stopWPIdx > nearestWPIdx ):
		endWPIdx = stopWPIdx
		#rospy.logwarn("stop requested...")
        #if( ( self.stopLightIdx != -1 or self.obstacleIdx != -1 ) and self.stopLightIdx > nearestWPIdx ):
        #	endWPIdx = self.stopLightIdx  #Only need to go to stop light
		
        # Create initial control/final waypoints
	for i in range(nearestWPIdx, endWPIdx ):
		cntrl_waypoints.waypoints.append(deepcopy(self.base_waypoints[i]))

    	#for idx in range(len(cntrl_waypoints.waypoints)-1,-1,-1):
	#	cntrl_waypoints.waypoints[idx].twist.twist.angular.z = 5.0 
        #        cntrl_waypoints.waypoints[idx].twist.twist.angular.x = 5.0 
        #        cntrl_waypoints.waypoints[idx].twist.twist.angular.y = 5.0 	
        #rospy.logdebug("start idx= %s, endWPidx = %s",nearestWPIdx, endWPIdx)
	#rospy.logdebug("control waypoings=%s", cntrl_waypoints)

	# decelerate if stop or obstacle...        
	#if self.stopLightIdx != -1 or self.obstacleIdx != -1:
	if stopWPIdx != -1:
		#rospy.logdebug("Stopping.....")		
		rospy.logwarn("Stopping.....Stoplight WP Index=%s", stopWPIdx)
		
		if nearestWPIdx < stopWPIdx:
			
 			stopD = self.distance(self.base_waypoints, nearestWPIdx, stopWPIdx )
			#rospy.logwarn("here dist = %s", stopD)
		else:
			stopD = 0

		if stopD < OPTIMAL_BREAKING_DIST:
			# Get max velocity from first waypoint
        		max_vel = cntrl_waypoints.waypoints[0].twist.twist.linear.x 
                        # Get incremental velocity value based on optimal breaking distance
        		vel_per_m = max_vel / OPTIMAL_BREAKING_DIST
              
			vel = 0.0
                        prevwp = None
                        bFirst = True
			# Set velocities for each control/final waypoint
			for idx in range(len(cntrl_waypoints.waypoints)-1,-1,-1):
				#rospy.logwarn("loop idx = %s length ctrlwp %s", idx, len(cntrl_waypoints.waypoints))
				if bFirst:		
								
					self.set_waypoint_velocity(cntrl_waypoints.waypoints, idx, vel)
					bFirst = False
				else:
					#Find distance between points
					
                                        pointD = self.distance(cntrl_waypoints.waypoints, idx, idx+1 )
                                        vel+= pointD * vel_per_m
					
					if vel > max_vel:
						vel = max_vel
					#rospy.logdebug("vel=%s", vel)
					self.set_waypoint_velocity(cntrl_waypoints.waypoints, idx, vel)                       
			

			#rospy.logdebug("control waypoings=%s", cntrl_waypoints.waypoints)

                
	
        # Publish to /final_waypoints
	self.final_waypoints_pub.publish(cntrl_waypoints)
        #rospy.logwarn("Before sleep")
	#rate.sleep()
        #rospy.logwarn("After")

	#rospy.logwarn("Published control %s waypoints", cntrl_waypoints.waypoints[0].pose.pose.position)

	#self.current_pose_sub.unregister()

        #pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement
	if( self.base_waypoints is None ):
		self.base_waypoints = waypoints.waypoints
 		self.lenBaseWPs = len(self.base_waypoints)
		#rospy.logdebug("base waypoints %s", self.base_waypoints)
	else:
		self.base_waypoints_sub.unregister()
        

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        #rospy.logdebug("traffic_cb. Msg.data=%s", msg.data)
        self.stopLightIdx = msg.data
        #self.stopLightIdx = -1        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later   	
        #rospy.logdebug("obstacle_cb. Msg=%s", msg.data)
	#self.obstacleIdx = msg.data
        self.obstacleIdx = -1

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def dist(self, p1, p2):
 	return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
