#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier
from path import computepath
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    
    # Get reference trajectory values
    q_ref = np.array(trajs[0](tcurrent))
    vq_ref = np.array(trajs[1](tcurrent))
    aq_ref = np.array(trajs[2](tcurrent))

    # Compute errors
    err_q = q - q_ref  # Position error
    err_vq = vq - vq_ref  # Velocity error

    print(f"t: {tcurrent:.2f}, max error: {np.max(np.abs(err_q)):.4f}")

    # Compute dynamics matrices
    # M(q)
    M = pin.crba(robot.model, robot.data, q)
    # h(q, vq) = C(q, vq)vq + g(q)
    nle = pin.nle(robot.model, robot.data, q, vq)

    # PD control law with feedforward (Inverse Dynamics)
    # aq_des = aq_ref - Kp * (q - q_ref) - Kv * (vq - vq_ref)
    aq_des = aq_ref - Kp * err_q - Kv * err_vq
    
    # Torque command: tau = M * aq_des + nle
    tau = M @ aq_des + nle

    # Convert to list for pybullet
    torques_list = tau.tolist()

    # Apply torques
    sim.step(torques_list)

    
def maketraj(path, T):
    print(f"Creating trajectory from path with {len(path)} points")
    
    # Extract all joint configurations (q) from the path
    q_points = [p[0] for p in path]
    
    if len(q_points) < 2:
        # Handle edge case: path is too short
        q_static = q_points[0] if q_points else robot.q0 # Failsafe
        control_points = [q_static, q_static, q_static]
    else:
        # Use waypoints as control points
        # Repeat start and end points 3 times to ensure 0 velocity and 0 acceleration
        q_start = q_points[0]
        q_end = q_points[-1]
        
        # Get intermediate points (if any)
        middle_points = q_points[1:-1]
        
        # Build control points list: [q_start, q_start, q_start, ...middle..., q_end, q_end, q_end]
        control_points = [q_start, q_start, q_start] + middle_points + [q_end, q_end, q_end]

    # Create the Bezier curve for joint positions
    q_of_t = Bezier(control_points, t_max=T)
    
    # Get the derivatives for velocity and acceleration
    vq_of_t = q_of_t.derivative(1)
    aq_of_t = vq_of_t.derivative(1) # aq_of_t is the same as vvq_of_t
    
    return q_of_t, vq_of_t, aq_of_t

    
if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    from setup_meshcat import updatevisuals
    
#     robot, sim, cube = setupwithpybullet()
    robot, sim, cube, viz = setupwithpybulletandmeshcat()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)
    
    if not successinit or not successend:
        print("Error: Could not compute grasp poses")
        exit()
        
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)#includes tuples (q,cube)

    if not path:
        print("Error: Path computation failed, no path found.")
        exit()

    # Setting initial configuration
    sim.setqsim(q0)

    # Create trajectory
    total_time = 4.
    trajs = maketraj(path, total_time)   
    tcur = 0.

    print("Starting simulation... Press Ctrl+C to stop.")
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        updatevisuals(viz, robot, cube, trajs[0](tcur))
        tcur += DT
        
    print("Control loop completed successfully!")