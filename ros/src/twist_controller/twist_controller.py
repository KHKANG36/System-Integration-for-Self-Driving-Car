
import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,Car_Param): # car_param is a class
        # TODO: Implement
        self.yawcontroller = YawController(wheel_base= Car_Param.wheel_base, steer_ratio = Car_Param.steer_ratio, min_speed= Car_Param.min_speed, max_lat_accel= Car_Param.max_lat_accel, max_steer_angle=Car_Param.max_steer_angle)
        #self.sampling_rate = sampling_rate
        self.Car_Param = Car_Param
        self.pid = PID(kp = 1, ki=0.05, kd = 0.01, mn = Car_Param.decel_limit, mx=Car_Param.accel_limit)
        self.LPF_acc = LowPassFilter(tau=0.2, ts =0.02) 
        self.LPF_steer = LowPassFilter(tau=5, ts =0.5) 
   
    def reset(self):
        self.pid.reset()

    def control(self, twist_cmd, current_velocity, duration):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        linear_vel = abs(twist_cmd.twist.linear.x)
        angular_vel = twist_cmd.twist.angular.z
        vel_err = linear_vel - current_velocity.twist.linear.x
        
        next_steer = self.yawcontroller.get_steering(linear_vel, angular_vel, current_velocity.twist.linear.x)
        next_steer = self.LPF_steer.filt(next_steer)
        
        acc = self.pid.step(vel_err, duration)
        #acc = self.LPF_acc.filt(acc)

        if acc >0.0:
           throttle = acc
           brake = 0.0
        else:
           throttle =0.0
           decel = - acc
           if decel <self.Car_Param.brake_deadband:
              decel = 0.0

           brake = decel* (self. Car_Param.vehicle_mass +self.Car_Param.fuel_capacity*GAS_DENSITY)*self.Car_Param.wheel_radius

        return throttle, brake, next_steer
