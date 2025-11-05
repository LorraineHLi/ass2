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
        dist = distance(node[1], c_rand) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx

def add_edge_and_vertex(G, parent, c, q):
    G.append((parent, c, q))
    
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
            return last_valid_c, last_valid_q
        last_valid_c = c_interp
        last_valid_q = q_interp
    return c_end, last_valid_q

def valid_edge(robot, cube, c_new, c_goal, q_new, discretisationsteps):
    '''check if can connect c_new to c_goal'''   
    c_final, q_final = new_conf(robot, cube, c_new, c_goal, q_new, discretisationsteps)
    
    goal_reached = distance(c_final, c_goal) < 1e-3
    return goal_reached

def rrt(robot, cube, c_init, c_goal, qinit, qgoal, k, delta_c, discretisationsteps_newconf=5, discretisationsteps_validedge=10):
    G = [(None, c_init, qinit)]
    
    for iteration in range(k):
        #sample random cube placement
        q_rand, c_rand = sample_valid_config(robot, cube)
        if c_rand is None:
            continue
        
        #find nearest vertex
        nearest_idx = nearest_vertex(G, c_rand)
        parent_idx, c_near, q_near = G[nearest_idx]
                
        #extend towards random sample
        c_new, q_new = new_conf(robot, cube, c_near, c_rand, q_near, 
                               discretisationsteps_newconf, delta_c)
        
        #add new node to graph
        new_idx = len(G)
        add_edge_and_vertex(G, nearest_idx, c_new, q_new)
        
        #try to connect to goal
        if valid_edge(robot, cube, c_new, c_goal, q_new, discretisationsteps_validedge):
            print("RRT: PATH FOUND!")
            add_edge_and_vertex(G, new_idx, c_goal, qgoal)
            return G, True
    
    print("RRT: Path not found within iteration limit")
    return G, False

def getpath(G):
    '''reconstruct path from graph'''
    if not G:
        return [], []
    
    path = []
    cube_placements = []
    node = G[-1]  
    
    while node[0] is not None:
        path.insert(0, node[2])  
        cube_placements.insert(0, node[1]) 
        node = G[node[0]]  
    
    path.insert(0, G[0][2])
    cube_placements.insert(0, G[0][1])
    
    return path, cube_placements

def shortcut(robot, cube, path, cube_placements, delta_c, discretisationsteps):
    if len(path) <= 2:
        return path, cube_placements
    
    print(f"  [Shortcut]: Applying shortcut to path with {len(path)} nodes")
    
    # Make copies to avoid modifying while iterating
    new_path = path.copy()
    new_cube_placements = cube_placements.copy()
    
    changed = True
    while changed:
        changed = False
        for i in range(len(new_path)):
            for j in reversed(range(i+2, len(new_path))):  
                if valid_edge(robot, cube, new_cube_placements[i], new_cube_placements[j], 
                             new_path[i], discretisationsteps):
                    # Remove nodes between i and j
                    new_path = new_path[:i+1] + new_path[j:]
                    new_cube_placements = new_cube_placements[:i+1] + new_cube_placements[j:]
                    print(f"Shortcut: Removed {j-i-1} nodes between {i} and {j}")
                    changed = True
                    break  # Restart the search since we modified the lists
            if changed:
                break
    
    print(f"Shortcut: Final path has {len(new_path)} configurations")
    return new_path, new_cube_placements

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
        path, cube_placements = getpath(G)
        print(f"Found path with {len(path)} configurations")
#         path, cube_placements = shortcut(robot, cube, path, cube_placements, delta_c, discretisationsteps_validedge)
#         print(f"After shortcut: {len(path)} configurations")
        
        return path, cube_placements
    else:
        print("No path found")
        return [],[]
    
    #return [qinit, qgoal]
    #pass


def displaypath(robot,path,dt,viz):
    for i, q in enumerate(path):
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
    
    path,cube_placements = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path, dt=0.1*30,viz=viz) #you ll probably want to lower dt
    
