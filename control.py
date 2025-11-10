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
    #TODO 
    q_des = trajs[0](tcurrent)
    vq_des = trajs[1](tcurrent) 
    aq_des = trajs[2](tcurrent)

    q_des = np.array(q_des)
    vq_des = np.array(vq_des)
    aq_des = np.array(aq_des)

    err_q = q_des - q # Position error
    err_vq = vq_des - vq # Velocity error

    print(f"t: {tcurrent:.2f}, max error: {np.max(np.abs(err_q)):.4f}")

    #print(f"q size: {q.shape}, q_des size: {q_des.shape}")
    #print(f"bulletCtrlJointsInPinOrder: {sim.bulletCtrlJointsInPinOrder}")
    #print(f"Number of controlled joints: {len(sim.bulletCtrlJointsInPinOrder)}")

    M = pin.crba(robot.model, robot.data, q)
    nle = pin.nle(robot.model, robot.data, q, vq)
    #print(f"M shape: {M.shape}, nle shape: {nle.shape}")

    tau = M @ (aq_des + Kp * err_q + Kv * err_vq) + nle
    #print(f"tau shape: {tau.shape}")

    torques_list = [0.0] * len(sim.bulletCtrlJointsInPinOrder)

    # Iterate 0 to 14 (robot.model.nv = 15)
    for i in range(robot.model.nv):
        # Pinocchio velocity index 'i' (0-14) maps to joint ID 'i+1' (1-15)
        joint_id = i + 1 

        # Check if this Pinocchio joint ID is in the list PyBullet controls
        if joint_id in sim.bulletCtrlJointsInPinOrder:
            # Get the index in PyBullet's 15-element list
            pybullet_index = sim.bulletCtrlJointsInPinOrder.index(joint_id)
            # Apply the calculated torque (tau[i]) to the correct slot
            torques_list[pybullet_index] = float(tau[i])

    sim.step(torques_list)

    
    #torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
    #sim.step(torques)

    
def maketraj(path,T): #TODO compute a real trajectory !
    print(f"Creating trajectory from path with {len(path)} points")
    
    # Convert all path points to numpy arrays and ensure proper shape
    path_arrays = []
    for i, q in enumerate(path):
        q_arr = np.array(q, dtype=float)
        path_arrays.append(q_arr)
        print(f"Path point {i}: shape {q_arr.shape}")
    
    q_start =path_arrays[0]
    q_end = path_arrays[-1]
    
    print(f"Start config shape: {q_start.shape}")
    print(f"End config shape: {q_end.shape}")
    
    # Start with 3 copies of q_start to enforce 0 velocity and acceleration
    control_points = [q_start, q_start, q_start]
    if len(path_arrays) > 2:
        control_points.extend(path_arrays[1:-1])
    # Add 3 copies of q_end to enforce 0 velocity and acceleration at the end
    control_points.extend([q_end, q_end, q_end])
    
    print(f"Total control points: {len(control_points)}")
    for i, cp in enumerate(control_points):
        if not isinstance(cp, np.ndarray):
            print(f"Warning: control point {i} is not a numpy array, converting...")
            control_points[i] = np.array(cp, dtype=float)
    
    q_of_t = Bezier(control_points,t_max=T)
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = q_of_t.derivative(2)
        
    return q_of_t, vq_of_t, vvq_of_t

    
if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    from setup_meshcat import updatevisuals
    
#     robot, sim, cube = setupwithpybullet()
    robot, sim, cube, viz = setupwithpybulletandmeshcat()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    q_elements= [x[0] for x in path]

    if not path:
            print("Error: No path found")
            exit()

    print(f"Path has {len(path)} configurations")

    # Print first and last configurations for debugging
    print(f"First config: {path[0][0]}")

    #setting initial configuration
    sim.setqsim(q0)

    #TODO this is just a random trajectory, you need to do this yourself
    total_time=4.

    print(f"Creating trajectory with total time: {total_time}")


    trajs = maketraj(q_elements, total_time)   

    tcur = 0.

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        updatevisuals(viz, robot, cube, trajs[0](tcur))
        tcur += DT
    print("Control loop completed successfully!")
    