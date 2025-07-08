import numpy as np
from systems import UnicycleSystem

def main():
    sys = UnicycleSystem(dt=0.1)
    x = np.array([0.0, 0.0, 0.0])   # initial pose (x, y, theta)
    u = np.array([1.0, 0.1])        # constant velocity commands (v, omega)

    print("Initial state:", x)
    for i in range(10):  # simulate 10 steps
        x = sys.step(x, u)
        print(f"Step {i+1}: {x}")

if __name__ == "__main__":
    main()
