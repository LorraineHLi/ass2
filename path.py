#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm

from config import LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
import time

from inverse_geometry import computeqgrasppose
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from setup_meshcat import updatevisuals
from tools import setupwithmeshcat
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

'''
sample cube
computeqgrasppose to find configuration
G stores cube position with corresponding configuration grasping it, and parent like (parent,cube,q)
interpolate based on cube position, not robot configuration'''

def sample_valid_config(robot, cube):
    '''sample a valid configuration using inverse geometry'''
    max_attempts = 20
    
    #define a reachable bounding box
    BOUNDS_X = [0.2, 0.6]
    BOUNDS_Y = [-0.4, 0.4]
    BOUNDS_Z = [0.9, 1.2] 
    
    #sample cube placement
    for i in range(max_attempts):
        x = np.random.uniform(BOUNDS_X[0], BOUNDS_X[1])
        y = np.random.uniform(BOUNDS_Y[0], BOUNDS_Y[1])
        z = np.random.uniform(BOUNDS_Z[0], BOUNDS_Z[1])
        rz = np.random.uniform(-np.pi/2, np.pi/2)
        
        sampled_cube_placement = pin.SE3(pin.utils.rotate('z', rz), np.array([x, y, z]))
        
        q, success = computeqgrasppose(robot, robot.q0, cube, sampled_cube_placement, None)
        if success:
            return q, sampled_cube_placement
    return None, None

def distance(c1, c2):    
    return np.linalg.norm(c1.translation - c2.translation)

def nearest_vertex(G, c_rand):
    '''returns the index of the Node of G with the cube position closest to c_rand'''
    min_dist = float('inf')
    idx = -1
    for (i, node) in enumerate(G):
        dist = distance(node[2], c_rand) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx

def add_edge_and_vertex(G, parent, q, c):
    G.append((parent, q, c))
    
def interpolate_cube(cube1, cube2, t):    
    '''interpolate between two cube placements'''
    #interpolate translation
    trans_interp = cube1.translation * (1 - t) + cube2.translation * t
    
    #interpolate rotation
    quat1 = pin.Quaternion(cube1.rotation)
    quat2 = pin.Quaternion(cube2.rotation)
    quat_interp = pin.Quaternion(quat1).slerp(t, quat2)
    
    return pin.SE3(quat_interp, trans_interp)

def new_conf(robot, cube, c_near, c_rand, q_near, discretisationsteps, delta_c=None):
    '''return the closest configuration c_new such that the path c_near => c_new is the longest
    along the linear interpolation (c_near,c_rand) that is collision free and of length <  delta_c'''
    c_end = c_rand.copy()
    dist = distance(c_near, c_rand)
    
    if delta_c is not None and dist > delta_c:
        c_end = interpolate_cube(c_near, c_rand, delta_c / dist)
    
    dt = 1.0 / discretisationsteps
    
    last_valid_c = c_near
    last_valid_q = q_near
    
    for i in range(1, discretisationsteps + 1):
        t = dt * i
        c_interp = interpolate_cube(c_near, c_end, t)
        
        q_interp, success = computeqgrasppose(robot, last_valid_q, cube, c_interp, None)
        if not success:
            return last_valid_q, last_valid_c
        last_valid_c = c_interp
        last_valid_q = q_interp
    return last_valid_q, c_end

def valid_edge(robot, cube, c_new, c_goal, q_new, discretisationsteps):
    '''check if can connect c_new to c_goal'''   
    q_final, c_final = new_conf(robot, cube, c_new, c_goal, q_new, discretisationsteps)
    
    goal_reached = distance(c_final, c_goal) < 1e-3
    if goal_reached:
        return True, q_final, c_final # Return True and the new q for the goal
    else:
        return False, None, None

def rrt(robot, cube, c_init, c_goal, qinit, qgoal, k, delta_c, discretisationsteps_newconf=5, discretisationsteps_validedge=10):
    G = [(None, qinit, c_init)]
    
    for iteration in range(k):
        #sample random cube placement
        q_rand, c_rand = sample_valid_config(robot, cube)
        if c_rand is None:
            continue
        
        #find nearest vertex
        nearest_idx = nearest_vertex(G, c_rand)
        parent_idx, q_near, c_near = G[nearest_idx]
                
        #extend towards random sample
        q_new, c_new = new_conf(robot, cube, c_near, c_rand, q_near, 
                               discretisationsteps_newconf, delta_c)
        
        #add new node to graph
        new_idx = len(G)
        add_edge_and_vertex(G, nearest_idx, q_new, c_new)
        
        #try to connect to goal
        is_valid, q_goal_new, c_final = valid_edge(robot, cube, c_new, c_goal, q_new, discretisationsteps_validedge)
        if is_valid:
            print("RRT: PATH FOUND!")
            # We add the original qgoal, as that is the desired target config
            add_edge_and_vertex(G, new_idx, q_goal_new, c_final) 
            return G, True
    
    print("RRT: Path not found within iteration limit")
    return G, False

def getpath(G):
    '''reconstruct path from graph'''
    if not G:
        return []
    
    path = []
    node = G[-1]  
    
    while node[0] is not None:
        # Store as (robot_config, cube_placement) tuple
        path.insert(0, (node[1], node[2]))  
        node = G[node[0]]  
    
    # Add start node
    path.insert(0, (G[0][1], G[0][2]))
    
    return path

def shortcut(robot, cube, path, delta_c, discretisationsteps):
    if len(path) <= 2:
        return path
    
    print(f"  [Shortcut]: Applying shortcut to path with {len(path)} nodes")
    
    # Make copies to avoid modifying while iterating
    new_path = path.copy()
    
    changed = True
    while changed:
        changed = False
        for i in range(len(new_path)): 
            for j in reversed(range(i+2, len(new_path))):  
                # Extract q and cube placement from the path tuples
                q_i, c_i = new_path[i]
                q_j, c_j = new_path[j]
                
                # Check if we can create a valid edge from i to j
                is_valid, q_j_new, c_final = valid_edge(robot, cube, c_i, c_j, q_i, discretisationsteps)
                
                if is_valid:
                    # Remove nodes between i and j
                    new_path = new_path[:i+1] + new_path[j:]
                    
                    # Update the robot config for the 'j' node (keep the same cube placement)
                    new_path[i+1] = (q_j_new, c_final)
                    
                    print(f"Shortcut: Removed {j-i-1} nodes between {i} and {j}. New length: {len(new_path)}")
                    changed = True
                    break  # Restart the search since we modified the lists
            if changed:
                break # Restart the outer 'while' loop
    
    print(f"Shortcut: Final path has {len(new_path)} configurations")
    return new_path

def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
    '''compute collision-free path from qinit to qgoal under grasping constraints'''
    
    #RRT parameters
    k = 300  
    delta_c = 0.3
    discretisationsteps_newconf = 7
    discretisationsteps_validedge = 20
        
    G, foundpath = rrt(robot, cube, cubeplacementq0, cubeplacementqgoal, 
                      qinit, qgoal, k, delta_c, 
                      discretisationsteps_newconf, discretisationsteps_validedge)
    
    if foundpath:
        path = getpath(G)
        print(f"Found path with {len(path)} configurations")
#         path = shortcut(robot, cube, path, delta_c, discretisationsteps_validedge)
#         print(f"After shortcut: {len(path)} configurations")
        
        return path
    else:
        print("No path found")
        return []
    
    #return [qinit, qgoal]
    #pass


def displaypath(robot,cube,path,dt,viz):
#     for q in path:
#         viz.display(q)
#         time.sleep(dt)


    print(f"Displaying path with {len(path)} configurations...")
    
    for (q, cube_placement) in path:
        setcubeplacement(robot, cube, cube_placement)
        updatevisuals(viz, robot, cube, q)
        time.sleep(dt)

if __name__ == "__main__":
    
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,cube,path,dt=0.5*5,viz=viz) #you ll probably want to lower dt
