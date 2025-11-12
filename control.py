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
from config import LEFT_HAND, RIGHT_HAND
    
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

GRASP_FORCE = 90.0  #squeeze force along world Y-axis
LIFT_FORCE = 45.0   #lifting force along world Z-axis

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    
    #get reference trajectory values
    q_ref = np.array(trajs[0](tcurrent))
    vq_ref = np.array(trajs[1](tcurrent))
    aq_ref = np.array(trajs[2](tcurrent))

    #compute errors
    err_q = q - q_ref
    err_vq = vq - vq_ref

    print(f"t: {tcurrent:.2f}, max position error: {np.max(np.abs(err_q)):.4f}")

    M = pin.crba(robot.model, robot.data, q)
    nle = pin.nle(robot.model, robot.data, q, vq)

    #desired acceleration
    aq_des = aq_ref - Kp * err_q - Kv * err_vq
    tau_motion = M @ aq_des + nle
    
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)

    #get Jacobians in world frame
    J_L = pin.getFrameJacobian(robot.model, robot.data, left_hand_id, pin.LOCAL_WORLD_ALIGNED)
    J_R = pin.getFrameJacobian(robot.model, robot.data, right_hand_id, pin.LOCAL_WORLD_ALIGNED)

    #6D force vectors [fx, fy, fz, tx, ty, tz]
    #left hand: squeezes in +Y, lifts in +Z
    f_L_vec = np.array([0, -GRASP_FORCE, LIFT_FORCE, 0, 0, 0])
    #right hand: squeezes in -Y, lifts in +Z
    f_R_vec = np.array([0, GRASP_FORCE, LIFT_FORCE, 0, 0, 0])

    #compute torques from forces: tau_force = J^T * f
    tau_force = J_L.T @ f_L_vec + J_R.T @ f_R_vec

    #final torque is motion + external forces 
    tau = tau_motion + tau_force

    torques_list = tau.tolist()
    sim.step(torques_list)
    
def maketraj(path, T):
    print(f"Creating trajectory from path with {len(path)} points")
    
    #extract all joint configurations (q) from the path
    q_points = [p[0] for p in path]
    
    #repeat start and end points 3 times to ensure 0 velocity and 0 acceleration
    q_start = q_points[0]
    q_end = q_points[-1]

    middle_points = q_points[1:-1]
    control_points = [q_start, q_start, q_start] + middle_points + [q_end, q_end, q_end]

    #Bezier curve for joint positions
    q_of_t = Bezier(control_points, t_max=T)
    
    #get the derivatives for velocity and acceleration
    vq_of_t = q_of_t.derivative(1)
    aq_of_t = vq_of_t.derivative(1) 
    
    return q_of_t, vq_of_t, aq_of_t

    
if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    from setup_meshcat import updatevisuals
    
    robot, sim, cube, viz = setupwithpybulletandmeshcat()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0, successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET, None)
    
    if not successinit or not successend:
        print("Error: Could not compute grasp poses")
        exit()
        
    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)#path includes tuples (q,cube)

    if not path:
        print("Error: Path computation failed, no path found.")
        exit()

    #setting initial configuration
    sim.setqsim(q0)

    #create trajectory
    total_time = 4.
    trajs = maketraj(path, total_time)   
    tcur = 0.

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        updatevisuals(viz, robot, cube, trajs[0](tcur))
        tcur += DT
        
    print("Control loop completed successfully!")