#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

import time
from scipy.optimize import fmin_bfgs
from setup_meshcat import updatevisuals

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    #TODO implement
    
    handidL = robot.model.getFrameId(LEFT_HAND)
    handidR = robot.model.getFrameId(RIGHT_HAND)

    oMhookL = getcubeplacement(cube, LEFT_HOOK)
    oMhookR = getcubeplacement(cube, RIGHT_HOOK)

    
    def cost(q):
        qProj = projecttojointlimits(robot, q)
        pin.framesForwardKinematics(robot.model, robot.data, qProj)
        oMhandL = robot.data.oMf[handidL]
        oMhandR = robot.data.oMf[handidR]
        errL = norm(pin.log(oMhandL.inverse() * oMhookL).vector) ** 2
        errR = norm(pin.log(oMhandR.inverse() * oMhookR).vector) ** 2
        
        return errL + errR
    
    def callback(q):
        if viz:
            qProj = projecttojointlimits(robot, q)
            updatevisuals(viz, robot, cube, qProj)
            time.sleep(1e-2)
    
    #optimisation
    qOpt = fmin_bfgs(cost, qcurrent, callback=callback, disp=False)
    qFinal = projecttojointlimits(robot, qOpt)
    
    #recalculate cost
    GRASP_TOLERANCE = 1e-3

    pin.framesForwardKinematics(robot.model, robot.data, qFinal)
    oMhandL = robot.data.oMf[handidL]
    oMhandR = robot.data.oMf[handidR]
    grasp_error = norm(pin.log(oMhandL.inverse() * oMhookL).vector) ** 2 + \
                  norm(pin.log(oMhandR.inverse() * oMhookR).vector) ** 2
    
    success = (grasp_error <= GRASP_TOLERANCE) and (not collision(robot, qFinal))
    
    #print ("TODO: implement me")
    return qFinal, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    
    updatevisuals(viz, robot, cube, q0)
 