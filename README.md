# README.md

No other software is needed apart from what is listed in `requirements.txt`.


## inverse_geometry.py
BFGS optimisation is used to minimise error between the robot’s hand poses and the cube’s grasping hooks, therefore solve the inverse geometry problem. 

Joint limits are enforced, and collision checks ensure feasible configurations.

## path.py
Following the algorithm taught in tutorial, RRT is used for path planning, which samples valid cube placements within a reachable space, and computes valid robot grasp configurations via `computeqgrasppose`. The planner interpolates cube translations linearly, and searches for a valid path to the goal.

We implemented nearest-neighbour search, and attempted path shortcutting (local optimisation to remove redundant nodes). The `shortcut` was partially implemented then disabled due to issues handling SE(3) frames.

## control.py
Bezier curve is used for joint trajectory generation from the RRT planned path. Motion is tracked with a computed-torque control law. The controller combines inverse dynamics with PD feedback, and additional forces are applied through the hand Jacobians simulate grasping and lifting the cube.