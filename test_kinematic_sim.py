import numpy as np
from systems import UnicycleSystem
from controllers import kinematic_controller
import matplotlib.pyplot as plt

def main():
    sys = UnicycleSystem(dt=0.1)
    x = np.array([0.0, 0.0, 0.0])             # initial state
    x_ref = np.array([1.0, 1.0, np.pi/4])     # goal state
    gains = (1.0, 2.0, -0.5)                   # controller gains

    trajectory = [x.copy()]
    controls = []

    for i in range(50):
        u = kinematic_controller(x, x_ref, gains)
        x = sys.step(x, u)
        trajectory.append(x.copy())
        controls.append(u)

    trajectory = np.array(trajectory)
    controls = np.array(controls)

    # Plot trajectory
    plt.figure(figsize=(8,8))
    plt.plot(trajectory[:,0], trajectory[:,1], label='Robot path')
    plt.plot(x_ref[0], x_ref[1], 'ro', label='Reference')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.title('Robot Trajectory using Kinematic Controller')
    plt.grid()
    plt.axis('equal')

    # Plot control inputs
    plt.figure()
    plt.plot(controls[:,0], label='Linear velocity v')
    plt.plot(controls[:,1], label='Angular velocity ω')
    plt.xlabel('Time step')
    plt.ylabel('Control input')
    plt.legend()
    plt.title('Control Inputs over Time')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()
