import numpy as np
from scipy.optimize import leastsq

def conditionSamples(params, data, errorModel):
    return np.array([errorModel(params, sample) for sample in data.T]).T

def getResidual(params, data, costFunction):
    residualSamples = costFunction(params, *data)
    residual = np.sum(residualSamples**2)
    return residual, residualSamples

def optimize(initialparams, data, costFunction):
    params, exitco = leastsq(costFunction, initialparams, data, maxfev=500)
    residual, residualSamples = getResidual(params, data, costFunction)
    return params, residual, residualSamples
