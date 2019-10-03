import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from calibration.models import accelerometer, gyroscope, magnetometer
from calibration.recorder import progress
from calibration import staticdetector as sd
from calibration import signalhelpers as sig


class Calibrator():

    def __init__(self):
        self.numthresholds = 10
        self.threshmultmax = 100
        self.variancewindowsize = 0.5
        self.minduration = 1

    async def calibrate_accelerometer(self, accl, time):
        print("Computing static detector signal")
        sr = sig.getMeanSamplingRate(time)
        variancesignal = sd.getVarianceSignal(
            accl, 
            self.variancewindowsize,
            sr)
        minv = np.min(variancesignal)
        threshstep = (self.threshmultmax-1) / self.numthresholds
        threshmultipliers = np.arange(1, self.threshmultmax, threshstep)
        thresholds = minv * threshmultipliers

        optimalresidual = float('inf')

        print("Searching for optimal accelerometer calibration")

        for thresh in thresholds:
            staticsignal = sd.thresholdVarianceSignal(
                variancesignal, 
                thresh)
            intervals = sd.signalToIntervals(staticsignal)
            intervals = sd.removeBriefIntervals(
                intervals,
                self.minduration, sr)

            if len(intervals) >= 12:
                gravitysamples = np.array(
                    [np.mean(accl[:,i],1) for i in intervals]).T
                import warnings
                with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  params, residual = optimize(
                      accelerometer.initialparams, 
                      (gravitysamples,), 
                      accelerometer.costFunction)

                if residual < optimalresidual:
                    optimalthresh = thresh
                    optimalresidual = residual
                    optimalstaticintervals = intervals
                    optimalstaticsignal = staticsignal
                    optimalparams = params
                    optimalgravitysamples = gravitysamples

        if optimalresidual == float('inf'):
            print('unable to find enough static periods')
            print('plotting variance signal for your consideration')
            print('program will abort when plot is closed')
            plt.plot(variancesignal)
            plt.show()
            assert False, 'not enough static periods'

        final_static_intervals = optimalstaticintervals
        final_accl = conditionSamples(
            optimalparams, accl, 
            accelerometer.errorModel)
        final_gravity_samples = conditionSamples(
            optimalparams, optimalgravitysamples, 
            accelerometer.errorModel)

        print('accelerometer calibration successful')
        print('found {} static periods with optimal threshold of {}'
                .format(len(final_static_intervals), optimalthresh))
        print('plotting calibrated accelerometer data')
        print('close the plot to continue...')
        from matplotlib import collections
        fig, ax = plt.subplots()
        ax.set_title('Accelerometer')
        ax.plot(final_accl[0,:], 'r-', label='x\'')
        ax.plot(final_accl[1,:], 'r-', label='y\'')
        ax.plot(final_accl[2,:], 'r-', label='z\'')
        ax.plot(accl[0,:], 'b-', label='x')
        ax.plot(accl[1,:], 'b-', label='y')
        ax.plot(accl[2,:], 'b-', label='z')
        darkenstatic = collections.BrokenBarHCollection.span_where(
                np.arange(len(final_accl)), 
                ymin = np.min(final_accl),
                ymax = np.max(final_accl),
                where = optimalstaticsignal > 0, #TODO display thinning
                facecolor = 'black', alpha = 0.2,
                label = 'static')
        ax.add_collection(darkenstatic)
        ax.legend()
        plt.show()

        return optimalparams, final_accl, final_gravity_samples, final_static_intervals

    async def calibrate_magnetometer(self, magn, static_intervals, gravity_samples):
        print("Calibrating magnetometer")
        simpleparams = magnetometer.simpleParamEstimate(magn)
        samps = np.array([np.mean(magn[:,i], 1) for i in static_intervals]).T
        uncaldotresidual, _ = getResidual(
            magnetometer.initialparams, 
            (gravity_samples, samps), 
            magnetometer.costFunction)
        uncalnormresidual, _ = getResidual(
            magnetometer.initialparams, 
            (magn,), 
            magnetometer.costFunction2)
        simpledotresidual, _ = getResidual(
            simpleparams, 
            (gravity_samples, samps), 
            magnetometer.costFunction)
        simplenormresidual, _ = getResidual(
            simpleparams, 
            (magn,), 
            magnetometer.costFunction2)
        print(f"Initial dot residual           = {uncaldotresidual}")
        print(f"Initial norm residual          = {uncalnormresidual}")
        print(f"Simple estimates dot residual  = {simpledotresidual}")
        print(f"Simple estimates norm residual = {simplenormresidual}")
        print("Close initial plot to continue")
        magnetometer.plotDataset(magn)
        magnetometer.plotDataset(conditionSamples(simpleparams, magn, magnetometer.errorModel))

        print("Calibrating with dot product error model")
        dot1params, dot1dotresidual = optimize(
            magnetometer.initialparams, 
            (gravity_samples, samps), 
            magnetometer.costFunction)
        dot1normresidual, _ = getResidual(dot1params, (magn,), magnetometer.costFunction2)
        print(f"First pass dot residual  = {dot1dotresidual}")
        print(f"First pass norm residual = {dot1normresidual}")
        print("Done; close plot to continue")
        magnetometer.plotDataset(conditionSamples(dot1params, magn, magnetometer.errorModel))

        print("Calibrating with vector norm error model")
        normparams, normnormresidual = optimize(
            dot1params, 
            (magn,), 
            magnetometer.costFunction2)
        normdotresidual, _ = getResidual(normparams, (gravity_samples, samps), magnetometer.costFunction)
        print(f"Norm based dot residual = {normdotresidual}")
        print(f"Norm based norm residual = {normnormresidual}")
        print("Done; close plot to continue")
        magnetometer.plotDataset(conditionSamples(normparams, magn, magnetometer.errorModel))
        dps = magnetometer.dotProductSignal(normparams, gravity_samples, samps)
        plt.plot(dps)
        plt.show()
        print(np.mean(dps))

        print("Final pass with dot product error model")
        dot2params, dot2dotresidual = optimize(
            normparams, 
            (gravity_samples, samps, np.mean(dps)), 
            magnetometer.costFunction)
        dot2normresidual, _ = getResidual(dot2params, (magn,), magnetometer.costFunction2)
        print(f"Final pass dot residual = {dot2dotresidual}")
        print(f"Final pass norm residual = {dot2normresidual}")
        print("Done; close plot to continue")
        magnetometer.plotDataset(conditionSamples(dot2params, magn, magnetometer.errorModel))

        if dot1normresidual < normnormresidual: 
          print("Choosing first pass parameters")
          return dot1params
        elif normnormresidual < dot2normresidual:
          print("Choosing norm based parameters")
          return normparams
        else:
          print("Choosing final pass parameters")
          return dot2params

    def calibrateGyroscope(self, data):
        if self.indicate: print("calibrating gyroscope...")
        # normalize gravity samps, since we only care about their direction
        gravitysamples = np.array([samp/np.linalg.norm(samp) 
                                   for samp in self.gravitysamples])
        biasfreedata, bias = gyroscope.removeBias(data, 
                                                  self.staticintervals)

        return bias ################################### TODO: make the rest work
        preppeddata = gyroscope.prepData(self.gravitysamples,
                                         self.staticintervals,
                                         data)
        print(f" keeping {len(preppeddata)} prepped data from {len(self.staticintervals)-1} static intervals")

        params, residual, residualSamples = optimize(gyroscope.initialparams, 
                                    (preppeddata, biasfreedata, self.accldata),
                                    gyroscope.costFunction)

        self.gyroresidual = residual
        self.gyrodata = conditionSamples(params, 
                                         biasfreedata, 
                                         gyroscope.errorModel)

        if self.indicate:
            print('gyroscope calibration successful')
            print('plotting calibrated gyroscope data')
            print('close the plot to continue...')
            fig, ax = plt.subplots()
            ax.set_title('Gyrodata')
            ax.plot(self.gyrodata[:,0], label='x')
            ax.plot(self.gyrodata[:,1], label='y')
            ax.plot(self.gyrodata[:,2], label='z')
            ax.legend()
            plt.show()
            plt.plot(self.gyrodata)
            plt.show()

        return params, bias
