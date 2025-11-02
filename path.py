#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm

from config import LEFT_HAND, RIGHT_HAND
import time

from inverse_geometry import computeqgrasppose

def lerp(q0, q1, t):    
    return q0 * (1 - t) + q1 * t

def distance(q1, q2):    
    return np.linalg.norm(q2 - q1)

def sample_valid_config(robot, cube, current_cube_placement):
    """Sample a valid configuration using inverse geometry"""
    max_attempts = 20
    for _ in range(max_attempts):
        # Sample random cube placement
        trans_perturbation = np.random.uniform(-0.2, 0.2, 3)
        rot_perturbation = pin.exp3(np.random.uniform(-0.3, 0.3, 3))
        
        new_trans = current_cube_placement.translation + trans_perturbation
        new_rot = current_cube_placement.rotation @ rot_perturbation
        
        sampled_cube_placement = pin.SE3(new_rot, new_trans)
        
        # Compute inverse geometry for this cube placement
        q, success = computeqgrasppose(robot, robot.q0, cube, sampled_cube_placement)
        
        if success:
            return q, sampled_cube_placement
    
    return None, None

def interpolate_se3(se3_1, se3_2, t):
    """Interpolate between two SE3 poses"""
    # Interpolate translation
    trans_interp = lerp(se3_1.translation, se3_2.translation, t)
    
    # Interpolate rotation using SLERP
    # Convert to quaternions for proper interpolation
    quat1 = pin.Quaternion(se3_1.rotation)
    quat2 = pin.Quaternion(se3_2.rotation)
    quat_interp = pin.Quaternion(quat1).slerp(t, quat2)
    
    return pin.SE3(quat_interp, trans_interp)

def interpolate_with_grasping(robot, cube, q0, q1, cube_placement0, cube_placement1, discretization_steps=5):
    """Interpolate cube placement and find valid robot configurations for each step"""
    path = [q0]
    
    for i in range(1, discretization_steps):
        t = i / discretization_steps
        
        # Interpolate cube placement using proper SE3 interpolation
        cube_placement_interp = interpolate_se3(cube_placement0, cube_placement1, t)
        
        q_interp, success = computeqgrasppose(robot, path[-1] if path else q0, cube, cube_placement_interp)
        if not success:
            return path if path else None
        
        path.append(q_interp)
    path.append(q1)
    return path

def nearest_vertex(G, q_rand):
    """Find nearest vertex in graph to random sample"""
    configs = np.array([node[1] for node in G])
    distances = np.linalg.norm(configs - q_rand, axis=1)
    return np.argmin(distances)

def new_conf(robot, cube, q_near, q_rand, cube_placement_near, delta_q, discretization_steps):
    """Extend from q_near towards q_rand"""
    dist = distance(q_near, q_rand)
    
    if dist <= delta_q:
        # Use the sampled configuration directly
        q_new, cube_placement_new = sample_valid_config(robot, cube, cube_placement_near)
    else:
        # Move by delta_q towards q_rand in configuration space
        direction = (q_rand - q_near) / dist
        q_target = q_near + direction * delta_q
        
        # Sample a configuration near this target
        q_new, cube_placement_new = sample_valid_config(robot, cube, cube_placement_near)
        if q_new is None:
            return None, None
    
    # Check path validity with small steps
    if q_new is not None:
        # Use interpolation to check path validity
        path_segment = interpolate_with_grasping(
            robot, cube, q_near, q_new, cube_placement_near, cube_placement_new, 5
        )
        if path_segment is None or len(path_segment) == 0:
            return None, None
    
    return q_new, cube_placement_new

def valid_edge(robot, cube, q_new, q_goal, cube_placement_new, cube_placement_goal, discretization_steps):
    """Check if we can connect directly to goal using interpolation"""
    path = interpolate_with_grasping(
        robot, cube, q_new, q_goal, cube_placement_new, cube_placement_goal, discretization_steps
    )
    return path is not None and len(path) > 0

def getpath(G):
    """Reconstruct path from graph"""
    if not G:
        return []
    
    path = []
    
    # Start from goal node (last node)
    current_idx = len(G) - 1
    
    while current_idx is not None:
        parent_idx, q, cube_placement = G[current_idx]
        path.insert(0, q)
        current_idx = parent_idx
    
    return path

def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    """Compute collision-free path from qinit to qgoal under grasping constraints"""
    
    # RRT parameters
    k = 300  # Reduced iterations for testing
    delta_q = 0.3  # Step size
    discretization_steps = 3
    
    # Initialize graph with start node
    G = [(None, qinit, cubeplacementq0)]
    
    for iteration in range(k):
        # Sample random configuration
        if np.random.random() < 0.1:
            q_rand, cube_placement_rand = qgoal, cubeplacementqgoal
        else:
            q_rand, cube_placement_rand = sample_valid_config(robot, cube, cubeplacementq0)
            
        if q_rand is None:
            continue
        
        # Find nearest vertex
        nearest_idx = nearest_vertex(G, q_rand)
        if nearest_idx == -1:
            continue
            
        q_near, cube_placement_near = G[nearest_idx][1], G[nearest_idx][2]
        
        # Extend towards random sample
        q_new, cube_placement_new = new_conf(
            robot, cube, q_near, q_rand, cube_placement_near, 
            delta_q, discretization_steps
        )
        
        if q_new is not None:
            # Add new node to graph
            new_idx = len(G)
            G.append((nearest_idx, q_new, cube_placement_new))
            
            # Try to connect to goal
            if valid_edge(robot, cube, q_new, qgoal, cube_placement_new, 
                         cubeplacementqgoal, discretization_steps):
                # Add goal node
                G.append((new_idx, qgoal, cubeplacementqgoal))
                print(f"Path found after {iteration} iterations!")
                path = getpath(G)
                
                # Refine the path using interpolation between all nodes
                refined_path = []
                for i in range(len(path) - 1):
                    q_start = path[i]
                    q_end = path[i + 1]
                    # Find corresponding cube placements in graph
                    cube_start = None
                    cube_end = None
                    for node in G:
                        if np.allclose(node[1], q_start):
                            cube_start = node[2]
                        if np.allclose(node[1], q_end):
                            cube_end = node[2]
                    
                    if cube_start is None:
                        cube_start = cubeplacementq0
                    if cube_end is None:
                        cube_end = cubeplacementqgoal
                    
                    segment = interpolate_with_grasping(
                        robot, cube, q_start, q_end, cube_start, cube_end, discretization_steps
                    )
                    if segment is None:
                        # If interpolation fails between graph nodes, use direct connection
                        refined_path.extend([q_start, q_end])
                    else:
                        refined_path.extend(segment[:-1])  # Avoid duplicates
                
                refined_path.append(path[-1])  # Add final configuration
                return refined_path
    
    print("Path not found within iteration limit")
    # Fallback: try direct interpolation
    direct_path = interpolate_with_grasping(
        robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, discretization_steps
    )
    return direct_path if direct_path is not None else [qinit, qgoal]


    #return [qinit, qgoal]
    #pass


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.01,viz=viz) #you ll probably want to lower dt
    
