import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from aioconsole import ainput

from .data import *
from pyMIMU.utils.drawing import Artist, draw_loop
from pyMIMU.utils.osc import AsyncOSC
import pyMIMU.calibration.accelerometer as accelerometer
import pyMIMU.calibration.magnetometer as magnetometer
import pyMIMU.calibration.gyroscope as gyroscope
import pyMIMU.calibration.staticdetector as static_detector

parser = argparse.ArgumentParser(
        description='Calibrate 9DOF MIMU/MARG Sensors', 
        epilog='Ask Travis for full instructions.') # TODO set up a website with full instructions

parser.add_argument('-port', type=int, 
        help='UDP port on which to listen for osc data (/raw ax ay az gx gy gz mx my mz time)')
parser.add_argument('-l', '--loadname', type=str, 
        help='Specify a file name to load data from a previous recording.', default='')
parser.add_argument('-o', '--savename', type=str, 
        help='Specify a file name to save data which can be recalled later to resume calibrating or retrieve calibration constants.', default='')
parser.add_argument('-c', '--calibrate', action='store_true', 
        help='Instantly run calibration in case a file is loaded. Has no effective if -l is not given.')

class CalibrationArgs:
  def __init__(self):
    self.port = None
    self.savename = ''
    self.loadname = ''
    self.calibrate = False
  
args = CalibrationArgs()
parser.parse_args(namespace=args)
rawdata = RawData()
caldata = CalibratedData()
executor = ProcessPoolExecutor(max_workers=11) # should be set by static detector number of thresholds

async def main():

  if args.loadname:
    print("Loading file")
    load_file(args.loadname, rawdata, caldata)
    print("File loaded")
    if args.calibrate:
      print("Calibrating based on loaded file")
      await calibrate(executor, rawdata, caldata)

  # get port etc in case they weren't specified
  if args.port is None:
    args.port = await ainput("Enter the port number on which to listen for raw data. This can also be specified from the command line with the -port option\nport >> ")

  osc = AsyncOSC(
      args.port, 
      osc_handler, rawdata, 
      outip='192.168.0.255', outport=6006)
  osc.send("/state/calibrate")

  print("Starting osc server")
  await osc.setup()

  print("Starting GUI")
  running = Flag()
  running.set(True)
  gui = asyncio.create_task(
      draw_loop(running, (2,5), get_artists, rawdata, caldata))

  print("Enter c to pause recording, calibrate, and then resume recording")
  print("Enter q to stop recording, calibrate one final time, and then quit")
  print("Enter Q to quit without re-calibrating one last time")
  print("Enter t to transmit calibration constants to a mubone device")
  print("Enter s to save the data recorded so far")
  print("Enter l to load raw data from a file")
  print("Enter L to load a file and immediately calibrate based on it")
  if args.savename is not '': 
    print(f"Raw data will be saved to {args.savename}.json on quit")
    print(f"Run the program again with -o {args.savename} to reload this data later")

  while True:

    user_input = await ainput("Enter c, q, Q, s, l, or L >> ")
    print("Got input", user_input)

    if user_input == 'c' or user_input == 'C':
      print("Calibrating...")
      osc.stop()
      await calibrate(executor, rawdata, caldata)
      osc.resume()
      continue

    elif user_input == 'q':
      print("Stopping after final calibration pass...")
      osc.stop()
      await calibrate(executor, rawdata, caldata)
      break
    
    elif user_input == 'Q':
      print("Quitting without calibrating...")
      osc.stop()
      break

    elif user_input == 's':
      if not args.savename:
        args.savename = await ainput("Enter a file name to save to. This can also be specified on the command line with the -o or --savename option.\nsavename >> ")
      print("Saving file...")
      save_file(args.savename, rawdata, caldata)
      print("File saved.")
      continue

    elif user_input == 't':
      await transmit(osc, caldata)
      continue

    elif user_input == 'l' or user_input == 'L':
      if not args.loadname:
        args.loadname = await ainput("Enter a file name to load from. This can also be specified on the command line with the -l or -loadname option.\nloadname >> ")
      print("Loading file...")
      load_file(args.loadname, rawdata, caldata)
      print("File loaded.")
      if user_input == 'L':
        await calibrate(executor, rawdata, caldata)
      continue


  # TODO: send these to the device as well as printing them

  if args.savename != '': 
    print("Saving file")
    save_file(args.savename, rawdata, caldata)
    print("File saved")

  running.set(False)
  await asyncio.sleep(0.1)
  if not gui.done():
    gui.cancel()
  try:
    await gui
  except asyncio.CancelledError:
    pass
  print("Done")
  return

async def calibrate(executor, rawdata, caldata):
  continue_flag = Flag()
  continue_flag.set(True)
  await calibrate_accelerometer(continue_flag, executor, rawdata, caldata)
  if not continue_flag: return
  await asyncio.gather(
      asyncio.create_task(calibrate_magnetometer(continue_flag, executor, rawdata, caldata)), 
      asyncio.create_task(calibrate_gyroscope(continue_flag, executor, rawdata, caldata))
  )
  accelerometer.printparams(caldata.accl_params)
  gyroscope.printparams(caldata.gyro_params, caldata.gyro_bias)
  magnetometer.printparams(caldata.magn_params)
  accelerometer.printparams_cpp(caldata.accl_params)
  gyroscope.printparams_cpp(caldata.gyro_params, caldata.gyro_bias)
  magnetometer.printparams_cpp(caldata.magn_params)
  max_accl_error = np.max(np.abs(caldata.accl_residual))
  avg_accl_error = np.mean(np.abs(caldata.accl_residual))
  avg_accl_norm  = np.mean([np.linalg.norm(g) for g in caldata.gravity_samples.T])
  max_magn_error = np.max(np.abs(caldata.magn_residual))
  avg_magn_error = np.mean(np.abs(caldata.magn_residual))
  avg_magn_norm  = np.mean([np.linalg.norm(h) for h in caldata.magn.T])
  print(f"Maximum accelerometer error: {max_accl_error}")
  print(f"Average accelerometer error: {avg_accl_error}")
  print(f"Average norm of gravity: {avg_accl_norm}")
  print(f"Maximum magnetometer error: {max_magn_error}")
  print(f"Average magnetometer error: {avg_magn_error}")
  print(f"Average norm of earth's magnetic field: {avg_magn_norm}")
  print(f"Sending calibration to device")

async def transmit(osc, caldata):
  avector, amatrix = accelerometer.params_message(caldata.accl_params)
  mvector, mmatrix = magnetometer.params_message(caldata.magn_params)
  gvector, gmatrix = gyroscope.params_message(caldata.gyro_params, caldata.gyro_bias)
  for m in [avector, amatrix, mvector, mmatrix, gvector, gmatrix]:
    osc.send(*m)

def osc_handler(address, refs, *args) -> None:
  rawdata = refs[0]
  if len(args) is not 10: 
    print(f"Warning: OSC got {address} with length {len(args)}, expected 10")
    return
  for a in args:
    if type(a) is not float:
      print("Warning: OSC got non-float argument")
      return
  time = args[9]
  if len(rawdata.time) > 0 and time == rawdata.time[-1]: return
  rawdata.raw.append(args)
  rawdata.accl.append(args[0:3])
  rawdata.gyro.append(args[3:6])
  rawdata.magn.append(args[6:9])
  rawdata.time.append(time)
  alat, alon = vecToCoords(rawdata.accl[-1][0], rawdata.accl[-1][1], rawdata.accl[-1][2])
  rawdata.accl_longitude.append(alon)
  rawdata.accl_latitude.append(alat)
  rawdata.accl_max = max(rawdata.accl_max, max(np.abs(rawdata.accl[-1])))
  if not rawdata.magn_deduped or rawdata.magn[-1] != rawdata.magn_deduped[-1]: 
    rawdata.magn_deduped.append(rawdata.magn[-1])
    mlat, mlon = vecToCoords(rawdata.magn[-1][0], rawdata.magn[-1][1], rawdata.magn[-1][2])
    rawdata.magn_longitude.append(mlon)
    rawdata.magn_latitude.append(mlat)
    rawdata.magn_max = max(rawdata.magn_max, max(np.abs(rawdata.magn[-1])))

async def calculate_variance(continue_flag, executor, rawdata, caldata, wait=1):
  samples = 6000
  while continue_flag:
    if len(rawdata) <= samples: 
      print(" Static interval detector requires more data ")
      continue_flag.set(False)
      break

    print(" Calculating static intervals ")
    static_future = executor.submit(static_detector.get_all_static_intervals, rawdata.accl, rawdata.time)
    while not static_future.done():
      await asyncio.sleep(wait)
    caldata.variance_signal, caldata.all_static_intervals = static_future.result()
    break

async def calibrate_accelerometer(continue_flag, executor, rawdata, caldata, andvariance = True, wait=5):
  while continue_flag:
    if andvariance: await calculate_variance(continue_flag, executor, rawdata, caldata)
    interval_lengths = [len(s) for s in caldata.all_static_intervals]
    max_intervals = 0 if not interval_lengths else max(interval_lengths)
    if max_intervals < 12:
      print(f"Accelerometer calibration needs at least 12 static intervals! Found at most {max_intervals}")
      continue_flag.set(False)
      break

    print(" Calibrating accelerometer ")
    acclfutures = [executor.submit(accelerometer.calibrate, rawdata.accl, static_intervals)
        for static_intervals in caldata.all_static_intervals if len(static_intervals) >= 12]
    while not all([future.done() for future in acclfutures]): 
      await asyncio.sleep(wait)
    acclresults = [future.result() for future in acclfutures]

    (aresidual, caldata.accl, caldata.gravity_samples, 
        caldata.optimal_static_intervals, caldata.accl_params, 
        caldata.accl_residual) = acclresults[0]

    caldata.accl_latitude, caldata.accl_longitude = vecToCoords(caldata.accl[0], caldata.accl[1], caldata.accl[2])
    caldata.max_accl_residual = max(caldata.max_accl_residual, max(np.abs(caldata.accl_residual)))
    break

async def calibrate_magnetometer(continue_flag, executor, rawdata, caldata, wait=10):
  while continue_flag:
    if not caldata.optimal_static_intervals: 
      print(" Magnetometer waiting for accelerometer calibration to finish ")
      await asyncio.sleep(wait)
      continue

    print(" Calibrating magnetometer ")
    magnfuture = executor.submit(magnetometer.calibrate, rawdata.magn, caldata.gravity_samples, caldata.optimal_static_intervals)
    while not magnfuture.done(): 
      await asyncio.sleep(wait)
    magnresults = magnfuture.result()
    mresidual, caldata.magn, caldata.magn_params, caldata.magn_residual = magnresults
    caldata.magn_latitude, caldata.magn_longitude = vecToCoords(caldata.magn[0], caldata.magn[1], caldata.magn[2])
    caldata.max_magn_residual = max(caldata.max_magn_residual, max(np.abs(caldata.magn_residual)))
    break

async def calibrate_gyroscope(continue_flag, executor, rawdata, caldata, wait=10):
  while continue_flag:
    if not caldata.optimal_static_intervals: 
      print(" Gyroscope waiting for accelerometer calibration to finish ")
      await asyncio.sleep(wait)
      continue

    print(" Calibrating gyroscope ")
    gyrofuture = executor.submit(gyroscope.calibrate, rawdata.gyro, caldata.gravity_samples, caldata.optimal_static_intervals)
    while not gyrofuture.done():
      await asyncio.sleep(wait)
    gyroresults = gyrofuture.result()
    gresidual, caldata.gyro, caldata.gyro_params, caldata.gyro_bias, caldata.gyro_residual = gyroresults
    caldata.max_gyro_residual = max(caldata.max_gyro_residual, max(np.abs(caldata.gyro_residual)))
    break

def get_artists(fig, axes, rawdata, caldata):
  return [ 
        Artist( # raw magnetometer latitude and longitude
          fig, axes[0][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "bo", 0.002,
          (lambda: rawdata.magn_longitude), (lambda: rawdata.magn_latitude))
      , Artist( # current raw magnetometer latitude and longitude
          fig, axes[0][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "m+", 1,
          (lambda: rawdata.magn_longitude[-1] if rawdata.magn_longitude else []), 
          (lambda: rawdata.magn_latitude[-1] if rawdata.magn_latitude else []))
      , Artist( # calibrated magnetometer latitude and longitude
          fig, axes[0][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "go", 0.002,
          (lambda: caldata.magn_longitude), (lambda: caldata.magn_latitude))

      , Artist( # raw magnetometer x y cross section
          fig, axes[0][1], (-1, 1), (-1, 1), "bo", 0.02,
          (lambda: [s[0]/rawdata.magn_max for s in rawdata.magn_deduped]), 
          (lambda: [s[1]/rawdata.magn_max for s in rawdata.magn_deduped]))
      , Artist( # raw magnetometer y z cross section
          fig, axes[0][2], (-1, 1), (-1, 1), "bo", 0.02,
          (lambda: [s[1]/rawdata.magn_max for s in rawdata.magn_deduped]), 
          (lambda: [s[2]/rawdata.magn_max for s in rawdata.magn_deduped]))
      , Artist( # raw magnetometer z x cross section
          fig, axes[0][3], (-1, 1), (-1, 1), "bo", 0.02,
          (lambda: [s[2]/rawdata.magn_max for s in rawdata.magn_deduped]), 
          (lambda: [s[0]/rawdata.magn_max for s in rawdata.magn_deduped]))

      , Artist( # current raw magnetometer x y cross section
          fig, axes[0][1], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.magn_deduped[-1][0]/rawdata.magn_max if rawdata.magn_deduped else []), 
          (lambda: rawdata.magn_deduped[-1][1]/rawdata.magn_max if rawdata.magn_deduped else []))
      , Artist( # current raw magnetometer y z cross section
          fig, axes[0][2], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.magn_deduped[-1][1]/rawdata.magn_max if rawdata.magn_deduped else []), 
          (lambda: rawdata.magn_deduped[-1][2]/rawdata.magn_max if rawdata.magn_deduped else []))
      , Artist( # current raw magnetometer z x cross section
          fig, axes[0][3], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.magn_deduped[-1][2]/rawdata.magn_max if rawdata.magn_deduped else []), 
          (lambda: rawdata.magn_deduped[-1][0]/rawdata.magn_max if rawdata.magn_deduped else []))

      , Artist( # calibrated magnetometer x y cross section
          fig, axes[0][1], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.magn[0,:] if caldata.magn.size > 0 else []), 
          (lambda: caldata.magn[1,:] if caldata.magn.size > 0 else []))
      , Artist( # calibrated magnetometer y z cross section
          fig, axes[0][2], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.magn[1,:] if caldata.magn.size > 0 else []), 
          (lambda: caldata.magn[2,:] if caldata.magn.size > 0 else []))
      , Artist( # calibrated magnetometer z x cross section
          fig, axes[0][3], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.magn[2,:] if caldata.magn.size > 0 else []), 
          (lambda: caldata.magn[0,:] if caldata.magn.size > 0 else []))

      , Artist( # accelerometer latitude and longitude
          fig, axes[1][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "bo", 0.002,
          (lambda: rawdata.accl_longitude), (lambda: rawdata.accl_latitude))
      , Artist( # current raw accelerometer latitude and longitude
          fig, axes[1][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "m+", 1,
          (lambda: rawdata.accl_longitude[-1] if rawdata.accl_longitude else []), 
          (lambda: rawdata.accl_latitude[-1] if rawdata.accl_latitude else []))
      , Artist( # calibrated accelerometer latitude and longitude
          fig, axes[1][0], (-np.pi, np.pi), (-np.pi/2, np.pi/2), "go", 0.002,
          (lambda: caldata.accl_longitude), (lambda: caldata.accl_latitude))

      , Artist( # raw accelerometer x y cross section
          fig, axes[1][1], (-1, 1), (-1, 1), "bo", 0.002,
          (lambda: [s[0]/rawdata.accl_max for s in rawdata.accl]), 
          (lambda: [s[1]/rawdata.accl_max for s in rawdata.accl]))
      , Artist( # raw accelerometer y z cross section
          fig, axes[1][2], (-1, 1), (-1, 1), "bo", 0.002,
          (lambda: [s[1]/rawdata.accl_max for s in rawdata.accl]), 
          (lambda: [s[2]/rawdata.accl_max for s in rawdata.accl]))
      , Artist( # raw accelerometer z x cross section
          fig, axes[1][3], (-1, 1), (-1, 1), "bo", 0.002,
          (lambda: [s[2]/rawdata.accl_max for s in rawdata.accl]), 
          (lambda: [s[0]/rawdata.accl_max for s in rawdata.accl]))

      , Artist( # current raw accelerometer x y cross section
          fig, axes[1][1], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.accl[-1][0]/rawdata.accl_max if rawdata.accl else []), 
          (lambda: rawdata.accl[-1][1]/rawdata.accl_max if rawdata.accl else []))
      , Artist( # current raw accelerometer y z cross section
          fig, axes[1][2], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.accl[-1][1]/rawdata.accl_max if rawdata.accl else []), 
          (lambda: rawdata.accl[-1][2]/rawdata.accl_max if rawdata.accl else []))
      , Artist( # current raw accelerometer z x cross section
          fig, axes[1][3], (-1, 1), (-1, 1), "m+", 1,
          (lambda: rawdata.accl[-1][2]/rawdata.accl_max if rawdata.accl else []), 
          (lambda: rawdata.accl[-1][0]/rawdata.accl_max if rawdata.accl else []))

      , Artist( # calibrated accelerometer x y cross section
          fig, axes[1][1], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.accl[0,:] if caldata.accl.size > 0 else []), 
          (lambda: caldata.accl[1,:] if caldata.accl.size > 0 else []))
      , Artist( # calibrated accelerometer y z cross section
          fig, axes[1][2], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.accl[1,:] if caldata.accl.size > 0 else []), 
          (lambda: caldata.accl[2,:] if caldata.accl.size > 0 else []))
      , Artist( # calibrated accelerometer z x cross section
          fig, axes[1][3], (-1, 1), (-1, 1), "go", 0.002,
          (lambda: caldata.accl[2,:] if caldata.accl.size > 0 else []), 
          (lambda: caldata.accl[0,:] if caldata.accl.size > 0 else []))

      , Artist( # accelerometer residual / cost function
          fig, axes[1][4], None, None, "r-", 1,
          (lambda: np.arange(len(caldata.accl_residual))), 
          (lambda: caldata.accl_residual/caldata.max_accl_residual))

      , Artist( # magnetometer residual / cost function
          fig, axes[0][4], None, None, "r-", 1,
          (lambda: np.arange(len(caldata.magn_residual))), 
          (lambda: caldata.magn_residual/caldata.max_magn_residual))
      ]

asyncio.run(main())
