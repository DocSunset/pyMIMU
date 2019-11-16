import numpy as np
from scipy.spatial.transform import Rotation as R
from functools import partial

from .optimization import *

initialparams = np.array([1, 0, 0, 
                          0, 1, 0, 
                          0, 0, 1, 

                          0, 0, 0])

initialparams2 = np.array([1, 0, 0, 
                              1, 0, 
                                 1, 

                           0, 0, 0])

def get_transform(params):
    # This model collapses soft-iron, non-orthogonality, and sensitivity errors
    # into a single linear matrix transformation, and collapses sensor bias and
    # hard-iron offset into a single bias term. 
    #   There are effectively no assumptions made about the nature of the sensor
    # distortions other than that they can be compensated with a linear map and
    # a translation, i.e. an affine transform
    #   This error model is very common for magnetometers.
    transform = np.array([ [params[0], params[1], params[2]],
                           [params[3], params[4], params[5]],
                           [params[6], params[7], params[8]]])
    bias = np.array([params[9], params[10], params[11]])
    return transform, bias

def get_transform_symmetric(params):
    # Like the default transform, all distortions are collapsed into a single
    # affine transform. Unlike the default transform, this model requires that
    # the linear map be symmetric. This has the effect that the transform which
    # compensates for the sensor errors will not rotate the sensor response
    # since rotations are necessarily non-symmetric. This may be considered
    # advantageous, especially when calibrating the magnetometer alone. In case
    # the cost function employed relates the magnetometer signal to another
    # sensor triad with which the magnetometer should be aligned, this error
    # model would be inappropriate since it is unable to rotate the
    # magnetometer response.
    transform = np.array([ [params[0], params[1], params[2]],
                           [params[1], params[3], params[4]],
                           [params[2], params[4], params[5]]])
    bias = np.array([params[6], params[7], params[8]])
    return transform, bias

def get_transform_rotation(params):
    # This error model is constrained to merely rotate the magnetometer, which
    # may be useful e.g. to align it with another sensor but otherwise prevent
    # the optimization algorithm from influencing its calibration.
    r = R.from_euler('xyz', params)
    return r.as_dcm(), np.array([0,0,0])

def errorModel(params, sample, getter=get_transform):
    transform, bias = getter(params)
    return np.matmul(transform, sample - bias)

def make_params(transform, bias):
    return np.array([
        transform[0,0], transform [0,1], transform[0,2],
        transform[1,0], transform [1,1], transform[1,2],
        transform[2,0], transform [2,1], transform[2,2],
        bias[0],        bias[1],         bias[2]])

def simpleParamEstimate(magn_sig, symmetric=False):
    minima = np.min(magn_sig,1)
    maxima = np.max(magn_sig,1)
    simplebias = (maxima + minima)/2.0
    simplescale = 2.0/(maxima - minima)
    if symmetric:
      return np.array([simplescale[0], 0, 0,
                          simplescale[1], 0,
                             simplescale[2],
                     simplebias[0], simplebias[1], simplebias[2]])
    else:
      return np.array([simplescale[0], 0, 0,
                     0, simplescale[1], 0,
                     0, 0, simplescale[2],
                     simplebias[0], simplebias[1], simplebias[2]])

def costFunctionDot(params, grav_samps, magn_samps, dotprod = 1.0, model=errorModel):
    # The cost is the sum of the difference between the dot product of the
    # earths magnetic field and gravity (a known constant), and the dot product
    # of the measurements of these two fields taken at the same moment in time.
    # Minimizing this cost function should result in all magnetometer errors
    # being compensated as well as aligning the magnetometer with the
    # accelerometer, but it relies on low-to-no-noise calibrated measurements
    # from the accelerometer, and potentially propagates errors from the
    # accelerometer into the magnetometer calibration.
    assert grav_samps.shape == magn_samps.shape, f"{grav_samps.shape}, {magn_samps.shape}"
    cal_samps = conditionSamples(params, magn_samps, model)
    return dotprod - dotProductSignal(grav_samps, cal_samps)

def dotProductSignal(grav_samps, magn_samps):
    return np.array([np.dot(grav, magn) for grav, magn in zip(grav_samps.T, magn_samps.T)])

def costFunctionNorm(params, magn_sig, norm=1.0, model=errorModel):
    # The cost is the difference between the norm of the calibrated
    # magnetometer measurement and the expected constant norm of the earths
    # magnetic field. This cost function is commonly used for many tri-axial
    # sensors by subjecting the sensor to a constant field of known magnitude
    # (e.g. earth's gravity or magnetic field).
    return np.array([norm - np.linalg.norm(model(params, samp), 2) for samp in magn_sig.T])

def calibrate_simple(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    params = simpleParamEstimate(magn)
    residual, residualSamples = getResidual(params, (magn,), costFunctionNorm)
    magn = conditionSamples(params, magn, errorModel)
    return residual, magn, params, residualSamples

def calibrate_dot(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
    params, residual, residualSamples = optimize(
        initialparams, 
        (gravity_samples, samps), 
        costFunctionDot)
    magn = conditionSamples(params, magn, errorModel)
    residual, residualSamples = getResidual(params, (magn,), costFunctionNorm)
    return residual, magn, params, residualSamples

def calibrate_norm(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
    params, residual, residualSamples = optimize(initialparams, (magn,), costFunctionNorm)
    magn = conditionSamples(params, magn, errorModel)
    return residual, magn, params, residualSamples

def calibrate(magn, gravity_samples, static_intervals):
    magn = np.array(magn).T
    samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
    simpleparams = simpleParamEstimate(magn, symmetric=True)
    normparams, _, _ = optimize(
        simpleparams, 
        (magn, 1.0, partial(errorModel, getter=get_transform_symmetric)),
        costFunctionNorm)

    def errormodel(params, sample):
      return errorModel(
          params, 
          errorModel(normparams, sample, get_transform_symmetric), 
          get_transform_rotation)

    rotation, residual, residualSamples = optimize(
        [0,0,0], 
        (gravity_samples, samps, 1.0, errormodel),
        costFunctionDot)

    transform, bias = get_transform_symmetric(normparams)
    transform = np.matmul(R.from_euler('xyz',rotation).as_dcm(), transform)
    params = make_params(transform, bias)
    residual, residualSamples = getResidual(params, (magn,), costFunctionNorm)
    magn = conditionSamples(params, magn, errorModel)
    return residual, magn, params, residualSamples

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
