# Assignment_new  
## Mobile Robot Tracking — Experiments A, B, C

---

## 📖 Overview
This project implements and compares three feedback control strategies for a non-holonomic 3-wheel robot (unicycle model):

1. **Kinematic Controller**  
   A closed-form polar-coordinate law:  
   \[
     v = k_\rho\,\rho,\quad
     \omega = k_\alpha\,\alpha + k_\beta\,\beta
   \]

2. **LQR Controller**  
   Discrete-time Linear Quadratic Regulator using the Riccati equation.

3. **MPC Controller**  
   Finite-horizon Model Predictive Control solved as a QP at each step.

We run three experiments:
- **A**: Sweep of kinematic gains  
- **B**: Sweep of LQR weight matrices (with/without input saturation)  
- **C**: Sweep of MPC horizons & cost weights  

---

## 📁 Repository Structure

rcognita-edu-main/
├── controllers.py # Kinematic, LQR, MPC implementations
├── systems.py # 3-wheel robot model & linearization
├── simulator.py # Closed-loop simulation engine
├── loggers.py # CSV data logger
├── utilities.py # Helpers (trajectory, angle wrap, etc.)
├── visuals.py # Plotting routines + CLI entrypoint
├── run_benchmark.py # Batch runner for Experiments A–C
├── logs/ # Generated CSV logs
├── figures/ # Generated result plots
├── report.pdf # ✅ Final report (if uploaded)
└── README.md # ← You are here

yaml
Copy code

---

## ▶️ Installation

```bash
git clone https://github.com/yourname/Assignment_new.git
cd rcognita-edu-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pandas
▶️ Usage
Run the experiments (this will generate CSV files in logs/):

bash
Copy code
python run_benchmark.py --dt 0.1 --T 20.0 --log_dir logs
Generate plots from the logs:

bash
Copy code
python visuals.py --log_dir logs --fig_dir figures
Inspect results:

CSV logs in logs/

Plots in figures/

📦 Experiment Reproducibility
All simulations automatically log data (robot states, velocities, tracking error) into CSV files using the loggers.py module. These logs are then parsed to generate the graphs seen in the report. Parameters for each controller can be modified in run_benchmark.py or by calling the controllers manually via PRESET_3wrobot_NI.py.

📸 Sample Output

Figure: Robot trajectories under different controllers.

📄 Final Report
The complete PDF report can be found here.

🧭 ROS & Gazebo Integration
The file 11zon.zip contains a ROS package to launch a TurtleBot3 simulation in Gazebo using your controllers.

To run:

bash
Copy code
unzip 11zon.zip
cd 11zon_pkg
catkin_make
source devel/setup.bash
roslaunch turtlebot3_control my_control.launch
🙏 Credits
This project is based on the rcognita-edu simulation framework by the THD Research Group.

📜 License
This repository is intended for academic and non-commercial use only.
