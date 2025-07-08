#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from scipy.linalg import solve_discrete_are
import tf


class TurtleBotLQR:
    def __init__(self):
        rospy.init_node("turtlebot_lqr_node")
        rospy.loginfo("LQR controller node launched.")

        # Load goal state
        self.target_state = np.array([
            rospy.get_param("~goal_x", 2.0),
            rospy.get_param("~goal_y", 3.0),
            rospy.get_param("~goal_theta", 0.5)
        ])

        # Time step for discrete system
        self.dt = 0.1

        # System matrices
        self.A = np.array([
            [1.0, 0.0, -self.dt * np.sin(self.target_state[2])],
            [0.0, 1.0,  self.dt * np.cos(self.target_state[2])],
            [0.0, 0.0, 1.0]
        ])

        self.B = np.array([
            [self.dt * np.cos(self.target_state[2]), 0.0],
            [self.dt * np.sin(self.target_state[2]), 0.0],
            [0.0, self.dt]
        ])

        self.Q = np.diag([5.0, 5.0, 1.0])
        self.R = np.diag([0.01, 0.1])

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        _, _, theta = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        current = np.array([pos.x, pos.y, theta])
        control = self.compute_lqr_control(current)

        cmd = Twist()
        cmd.linear.x = control[0]
        cmd.angular.z = control[1]
        self.cmd_pub.publish(cmd)

    def compute_lqr_control(self, state):
        # Solve DARE
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)

        # Compute LQR gain matrix
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        error = state - self.target_state
        error[2] = self.normalize_angle(error[2])  # Keep theta in [-pi, pi]

        u = -K @ error
        v, omega = u[0], u[1]

        # Optional saturation
        v = np.clip(v, -0.5, 0.5)
        omega = np.clip(omega, -1.0, 1.0)

        return v, omega

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


if __name__ == "__main__":
    try:
        controller = TurtleBotLQR()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
