import numpy as np

def get_all_static_intervals(accl, times):
  accl = np.array(accl).T
  times = np.array(times).T
  sampling_rate = getMeanSamplingRate(times)
  variance = variance_signal(accl, sampling_rate)
  thresholds = np.min(variance) * threshold_multipliers()
  all_static_intervals = []
  for threshold in thresholds:
    static_signal = (variance < threshold) * 1.0
    intervals = signalToIntervals(static_signal)
    intervals = removeBriefIntervals(intervals, sampling_rate)
    #intervals = removeLongIntervals(intervals, sampling_rate)
    all_static_intervals.append(intervals)
  return variance, all_static_intervals

# the signal is assumed to be more or less evenly sampled
def getMeanSamplingRate(times):
    intersampleperiods = np.diff(times)
    nonzeroperiods = intersampleperiods[np.where(intersampleperiods != 0)]
    rates = 1.0 / nonzeroperiods
    meansamplingrate = rates.mean()
    return meansamplingrate
    
def variance_signal(data, sampling_rate, windowsize=0.5):
    sizeinsamps = windowsize * sampling_rate
    length = len(data.T)
    if sizeinsamps % 1:
        sizeinsamps += 1
    halfsizeinsamps = sizeinsamps / 2
    currentsamp = 0
    variances = []
    while currentsamp < length:
        leftsamp = currentsamp - halfsizeinsamps
        rightsamp = currentsamp + halfsizeinsamps
        if leftsamp < 0: leftsamp = 0
        if rightsamp >= length: rightsamp = length - 1
        variance = [np.var(axis[int(leftsamp):int(rightsamp)])**2 for axis in data]
        variance = np.sum(variance)
        variances.append(variance)
        currentsamp += 1
    return variances

def threshold_multipliers(numthresholds = 10, threshmultmax = 100):
    threshstep = (threshmultmax-1) / numthresholds
    threshmultipliers = np.arange(1, threshmultmax, threshstep)
    return threshmultipliers

def signalToIntervals(sig):
    # sig is a static signal, i.e. 1 where static, 0 where dynamic
    indexarray = np.flatnonzero(sig)
    if len(indexarray) <= 0: return [], []
    staticinterval = [ indexarray[0] ]
    staticintervals = []

    for k in range(1,len(indexarray)):
        step = indexarray[k] - indexarray[k-1]
        if step > 1: 
            # k-1 is last step in a staticinterval, 
            staticintervals.append(np.asarray(staticinterval))
            staticinterval = []
        staticinterval.append(indexarray[k])

    if staticinterval != []: 
        staticintervals.append(np.asarray(staticinterval))
    return staticintervals

def removeBriefIntervals(intervals, samplingrate=1, minduration=1):
    minsamples = int(minduration*samplingrate)
    return [interval for interval in intervals if not len(interval) < minsamples]

def intervalsToSignal(intervals, length):
    indicies = np.asarray(intervals).reshape(-1)
    return np.isin(np.arange(length), indicies)
