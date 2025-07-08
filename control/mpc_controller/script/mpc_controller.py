#!/usr/bin/env python3

import rospy
import numpy as np
import casadi as ca
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf


class TurtleBotMPC:
    def __init__(self):
        rospy.init_node("turtlebot_mpc_node")
        rospy.loginfo("Starting Model Predictive Control node for TurtleBot...")

        # Goal state parameters
        self.target_pos = np.array([
            rospy.get_param("~goal_x", 5.0),
            rospy.get_param("~goal_y", 3.0),
            rospy.get_param("~goal_theta", 0.5)
        ])

        self.cmd_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.horizon = 10
        self.time_step = 0.1
        self.robot_radius = 0.2

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        _, _, current_theta = tf.transformations.euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        current_state = np.array([pos.x, pos.y, current_theta])

        v, omega = self.solve_mpc(current_state)

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(omega)
        self.cmd_publisher.publish(twist)

    def solve_mpc(self, x0):
        N = self.horizon
        dt = self.time_step

        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        states = ca.vertcat(x, y, theta)
        n_states = states.size()[0]

        v = ca.SX.sym("v")
        omega = ca.SX.sym("omega")
        controls = ca.vertcat(v, omega)
        n_controls = controls.size()[0]

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        f = ca.Function("f", [states, controls], [rhs])

        U = ca.SX.sym("U", n_controls, N)
        X = ca.SX.sym("X", n_states, N + 1)
        P = ca.SX.sym("P", n_states + N * n_states)

        Q = ca.diag([5.0, 5.0, 1.0])
        R = ca.diag([0.1, 0.1])

        obj = 0
        g = []

        X[:, 0] = P[0:3]

        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            ref = P[3 + k * 3: 3 + (k + 1) * 3]
            obj += ca.mtimes([(st - ref).T, Q, (st - ref)]) + ca.mtimes([con.T, R, con])
            st_next = X[:, k] + dt * f(X[:, k], U[:, k])
            X[:, k + 1] = st_next

        for k in range(N):
            g.append(X[0, k])
            g.append(X[1, k])

        opt_vars = ca.vertcat(ca.reshape(U, -1, 1))
        nlp = {'f': obj, 'x': opt_vars, 'p': P, 'g': ca.vertcat(*g)}

        solver = ca.nlpsol("solver", "ipopt", nlp)
        x_ref = np.tile(self.target_pos, N)

        init_controls = np.zeros((N, n_controls)).flatten()
        lbx = [-1.0] * len(init_controls)
        ubx = [1.0] * len(init_controls)

        lbg = [-ca.inf] * len(g)
        ubg = [ca.inf] * len(g)

        p_vector = np.concatenate((x0, x_ref))

        result = solver(x0=init_controls, p=p_vector, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

        optimal_u = result['x'].full().flatten()
        v_opt, omega_opt = optimal_u[0], optimal_u[1]
        return v_opt, omega_opt


if __name__ == "__main__":
    try:
        controller = TurtleBotMPC()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
