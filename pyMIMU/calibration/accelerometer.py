import numpy as np

from .optimization import *

initialparams = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

def getTransform(params):
    orthogonality = np.array([[1, params[0], params[1]], 
                              [0,         1, params[2]], 
                              [0,         0,         1]])
    scaling = np.array([[params[3], 0, 0],
                        [0, params[4], 0],
                        [0, 0, params[5]]])
    bias = np.array([params[6], params[7], params[8]])
    return np.matmul(orthogonality, scaling), bias

def errorModel(params, sample):
    transform, bias = getTransform(params)
    return np.matmul(transform, (sample - bias))

def costFunction(params, gravitysamples):
    # it is assumed, without loss of generality, that the norm of gravity == 1.0
    return np.array([1.0 - np.linalg.norm(errorModel(params, sample), 2) for sample in gravitysamples.T])

def calibrate(accl, intervals):
    if len(intervals) < 12: return None
    accl = np.array(accl).T
    gravitysamples = np.array([np.mean(accl[:,i],1) for i in intervals]).T
    import warnings
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      params, residual, residualSamples = optimize(
          initialparams, 
          (gravitysamples,), 
          costFunction)
    accl = conditionSamples(params, accl, errorModel)
    gravitysamples = conditionSamples(params, gravitysamples, errorModel)
    return residual, accl, gravitysamples, intervals, params, residualSamples

def params_message(params):
    transform, bias = getTransform(params)   
    vector = ('/calibration/accl/vector', float(bias[0]), float(bias[1]), float(bias[2]))
    matrix = ('/calibration/accl/matrix', 
        float(transform[0][0]), float(transform[0][1]), float(transform[0][2]), 
        float(transform[1][0]), float(transform[1][1]), float(transform[1][2]), 
        float(transform[2][0]), float(transform[2][1]), float(transform[2][2]))
    return vector, matrix

def printparams(params):
    transform, bias = getTransform(params)   
    print('/calibration/accl/vector {} {} {}'.format(bias[0], bias[1], bias[2]))
    print('/calibration/accl/matrix {} {} {} {} {} {} {} {} {}'.format(
      transform[0][0], transform[0][1], transform[0][2], 
      transform[1][0], transform[1][1], transform[1][2], 
      transform[2][0], transform[2][1], transform[2][2]))

def printparams_cpp(params):
    transform, bias = getTransform(params)   
    print(f'calibrator.cc.abias << {bias[0]}, {bias[1]}, {bias[2]};')
    print(f'calibrator.cc.acclcalibration << {transform[0][0]}, {transform[0][1]}, {transform[0][2]}, {transform[1][0]}, {transform[1][1]}, {transform[1][2]}, {transform[2][0]}, {transform[2][1]}, {transform[2][2]};')
