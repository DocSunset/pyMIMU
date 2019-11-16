from matplotlib import pyplot as plt
import sys
import numpy as np
from pyquaternion import Quaternion

from .staticdetector import signalToIntervals
from .optimization import *

initialparams = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
degtorad = 180/np.pi
radtodeg = np.pi/180

def getTransform(params):
    orthogonality = np.array([[1, params[0], params[1]], 
                              [params[2], 1, params[3]], 
                              [params[4], params[5], 1]])
    scaling = np.array([[params[6], 0, 0],
                        [0, params[7], 0],
                        [0, 0, params[8]]])
    return np.matmul(orthogonality, scaling)

def errorModel(params, sample):
    # assumes bias has already been removed from data
    transform = getTransform(params)
    conditionedsample = np.matmul(transform, sample[:3])
    return np.array([conditionedsample[0], 
                     conditionedsample[1], 
                     conditionedsample[2], 
                     sample[3]])

def quaternion_derivative(quat, rate):
    return 0.5 * quat * Quaternion(vector=rate)

def rk4step(quat, lastsamp, nextsamp):
    dt = nextsamp[3] - lastsamp[3]
    if dt == 0.0: return quat
    lastsamp = lastsamp[:3]
    nextsamp = nextsamp[:3]
    halfsamp = 0.5*(lastsamp + nextsamp)
    k1 = quaternion_derivative(quat, lastsamp)
    k2 = quaternion_derivative(quat + dt * k1 / 2.0, halfsamp)
    k3 = quaternion_derivative(quat + dt * k2 / 2.0, halfsamp)
    k4 = quaternion_derivative(quat + dt * k3, nextsamp)
    ret = quat + (dt/6.0) * (k1 + 2 * (k2 + k3) + k4)
    return ret.unit

def rk4integrate(interval, gyrodata, startsamp):
    #animation = [startsamp]
    rotationquat = Quaternion()
    for i in interval[:-1]:
        rotationquat = rk4step(rotationquat, gyrodata[i], gyrodata[i+1])
        #animation.append(rotationquat.rotate(startsamp))

    #animation = np.asarray(animation)
    #plt.plot(animation[:,0], 'r-')
    #plt.plot(animation[:,1], 'g-')
    #plt.plot(animation[:,2], 'b-')
    #plt.plot(accldata[interval][:,0], 'r:', alpha=0.5)
    #plt.plot(accldata[interval][:,1], 'g:', alpha=0.5)
    #plt.plot(accldata[interval][:,2], 'b:', alpha=0.5)
    #plt.show()
    return rotationquat

def linearintegrate(interval, gyrodata, startsamp):
    #animation = [startsamp]
    q = Quaternion()
    for i in interval[:-1]:
        dt = gyrodata[i+1,3] - gyrodata[i,3]
        if dt == 0.0: animation.append(q.rotate(startsamp)); continue
        q += dt * quaternion_derivative(q, gyrodata[i,:3])
        q = q.unit
        #animation.append(q.rotate(startsamp))
    #animation = np.asarray(animation)
    #plt.plot(animation[:,0], 'r-')
    #plt.plot(animation[:,1], 'g-')
    #plt.plot(animation[:,2], 'b-')
    #plt.plot(accldata[interval][:,0], 'r:', alpha=0.5)
    #plt.plot(accldata[interval][:,1], 'g:', alpha=0.5)
    #plt.plot(accldata[interval][:,2], 'b:', alpha=0.5)
    #plt.show()
    return q

def removeBias(data, staticintervals):
    biases = np.array([ np.mean(data[:,i],1) for i in staticintervals]).T
    bias = np.mean(biases, 1)
    biasfreedata = (data.T - bias).T
    return biasfreedata, bias

def prepData(gravity_samples, staticintervals, biasfreedata):
    data = []
    for i in np.arange(len(staticintervals)-1):
        assert len(staticintervals[i]) > 1
        # we want the dynamic interval, i.e. the interval between static intervals
        # i.e. the interval from the last index of the former static interval, up to the first index of the latter
        interval = np.arange(staticintervals[i][-1]+1,
                             staticintervals[i+1][0]+1)
        prevgrav = gravity_samples[i]
        nextgrav = gravity_samples[i+1]
        if np.linalg.norm(prevgrav - nextgrav) < 0.01: 
          print ("got rid of two very similar gravity samples")
          continue
        data.append( (prevgrav, nextgrav, interval) )
    return data

def costFunction(params, preppeddata, biasfreedata):
    conditioned_data = np.array([errorModel(params, sample) 
                            for sample in biasfreedata])
    ret = []
    for datatuple in preppeddata:
        prevgrav, nextgrav, interval = datatuple
        rotation = rk4integrate(interval, conditioned_data, prevgrav)
        estimate = rotation.rotate(prevgrav)
        esterror = np.linalg.norm(nextgrav - estimate)
        diff = nextgrav - estimate
        ret.append(esterror)
        #with np.printoptions(precision=3, suppress=True):
        #  print(f'prevgrav: {prevgrav}, nextgrav: {nextgrav}, rotation: {rotation.elements}, estimate: {estimate}, difference: {diff}, error: {esterror}')

    ret = np.asarray(ret)

    print(np.sqrt(np.sum(ret**2)))
    return ret

def calibrate(gyro, gravity_samples, static_intervals):
    # normalize gravity samps, since we only care about their direction
    gyro = np.array(gyro).T
    gravity_samples = np.array([samp/np.linalg.norm(samp) for samp in gravity_samples])
    biasfreedata, bias = removeBias(gyro, static_intervals)
    return 1, biasfreedata, initialparams, bias, np.array([1,2,3])
    
    ################################### TODO: make the rest work
    preppeddata = prepData(gravity_samples, static_intervals, gyro)
    params, residual, residualSamples = optimize(
        initialparams, 
        (preppeddata, biasfreedata),
        costFunction)
    gyro = conditionSamples(params, biasfreedata, errorModel)
    return residual, gyro, params, bias, residualSamples

def params_message(params, bias):
    transform = getTransform(params)   
    vector = ('/calibration/gyro/vector', float(bias[0]), float(bias[1]), float(bias[2]))
    matrix = ('/calibration/gyro/matrix', 
        float(transform[0][0]), float(transform[0][1]), float(transform[0][2]), 
        float(transform[1][0]), float(transform[1][1]), float(transform[1][2]), 
        float(transform[2][0]), float(transform[2][1]), float(transform[2][2]))
    return vector, matrix

def printparams(params, bias):
    transform = getTransform(params)   
    print(f'/calibration/gyro/vector {bias[0]} {bias[1]} {bias[2]}')
    print(f'/calibration/gyro/matrix {transform[0][0]} {transform[0][1]} {transform[0][2]} {transform[1][0]} {transform[1][1]} {transform[1][2]} {transform[2][0]} {transform[2][1]} {transform[2][2]} ')

def printparams_cpp(params, bias):
    transform = getTransform(params)   
    print(f'calibrator.cc.gbias << {bias[0]}, {bias[1]}, {bias[2]};')
    print(f'calibrator.cc.gyrocalibration << {transform[0][0]}, {transform[0][1]}, {transform[0][2]}, {transform[1][0]}, {transform[1][1]}, {transform[1][2]}, {transform[2][0]}, {transform[2][1]}, {transform[2][2]};')
    
