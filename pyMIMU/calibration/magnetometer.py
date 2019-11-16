import numpy as np
from matplotlib import pyplot as plt

from .optimization import *

initialparams = np.array([1, 0, 0, 
                          0, 1, 0, 
                          0, 0, 1, 

                          0, 0, 0])

def errorModel(params, sample):
    transform = np.array([ [params[0], params[1], params[2]],
                           [params[3], params[4], params[5]],
                           [params[6], params[7], params[8]]])
    bias = np.array([params[9], params[10], params[11]])
    return np.matmul(transform, sample - bias)

def simpleParamEstimate(magn_sig):
    minima = np.min(magn_sig,1)
    maxima = np.max(magn_sig,1)
    simplebias = (maxima + minima)/2.0
    simplescale = 2.0/(maxima - minima)
    return np.array([simplescale[0], 0, 0,
                     0, simplescale[1], 0,
                     0, 0, simplescale[2],

                     simplebias[0], simplebias[1], simplebias[2]])

def dotProductSignal(params, grav_samps, magn_samps):
    cal_samps = conditionSamples(params, magn_samps, errorModel)
    return np.array([np.dot(grav, magn) for grav, magn in zip(grav_samps.T, cal_samps.T)])

def costFunction(params, grav_samps, magn_samps, dotprod = 1.0):
    assert grav_samps.shape == magn_samps.shape, f"{grav_samps.shape}, {magn_samps.shape}"
    return dotprod - dotProductSignal(params, grav_samps, magn_samps)

def costFunction2(params, magn_sig):
    return np.array([1.0 - np.linalg.norm(errorModel(params, samp), 2) for samp in magn_sig.T])

def super_simple_calibrate(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    params = simpleParamEstimate(magn)
    residual, residualSamples = getResidual(params, (magn,), costFunction2)
    magn = conditionSamples(params, magn, errorModel)
    return residual, magn, params, residualSamples

def simple_calibrate(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
    simpleparams = simpleParamEstimate(magn)
    params, residual, residualSamples = optimize(
        simpleparams, 
        (gravity_samples, samps), 
        costFunction)
    magn = conditionSamples(params, magn, errorModel)
    residual, residualSamples = getResidual(params, (magn,), costFunction2)
    return residual, magn, params, residualSamples

def calibrate(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
    simpleparams = simpleParamEstimate(magn)
    normparams, _, _ = optimize(simpleparams, (magn,), costFunction2)
    dps = dotProductSignal(normparams, gravity_samples, samps)
    params, residual, residualSamples = optimize(
        normparams, 
        (gravity_samples, samps),#, np.mean(dps)), 
        costFunction)
    residual, residualSamples = getResidual(params, (magn,), costFunction2)
    magn = conditionSamples(params, magn, errorModel)
    return residual, magn, params, residualSamples

def plotDataset(magn_sig):
    x = magn_sig[0,:]
    y = magn_sig[1,:]
    z = magn_sig[2,:]
    latitude, longitude = vecToCoords(x,y,z)
    plt.plot(longitude, latitude, 'bo', alpha=0.02)
    plt.show()

def params_message(params):
    vector = ('/calibration/magn/vector', float(params[9]), float(params[10]), float(params[11]))
    matrix = ('/calibration/magn/matrix', 
      float(params[0]), float(params[1]), float(params[2]),
      float(params[3]), float(params[4]), float(params[5]),
      float(params[6]), float(params[7]), float(params[8]))
    return vector, matrix

def printparams(params):
    print('/calibration/magn/vector {} {} {}'.format(
      params[9], params[10], params[11]))
    print('/calibration/magn/matrix {} {} {} {} {} {} {} {} {}'.format(
      params[0], params[1], params[2],
      params[3], params[4], params[5],
      params[6], params[7], params[8]))

def printparams_cpp(params):
    print(f'calibrator.cc.mbias << {params[9]}, {params[10]}, {params[11]};')
    print(f'calibrator.cc.magncalibration << {params[0]}, {params[1]}, {params[2]}, {params[3]}, {params[4]}, {params[5]}, {params[6]}, {params[7]}, {params[8]};')
