import numpy as np
import json
from .accelerometer import initialparams as ap
from .gyroscope import initialparams as gp
from .magnetometer import initialparams as mp

class Flag:
  def __init__(self):
    self._flag = False

  def set(self, b):
    self._flag = bool(b)

  def __bool__(self):
    return self._flag

  def __nonzero__(self):
    return self._flag

class RawData:
  def __init__(self):
    self.raw = []
    self.accl = []
    self.gyro = []
    self.magn = []
    self.magn_deduped = []
    self.time = []
    self.accl_longitude = []
    self.accl_latitude  = []
    self.magn_longitude = []
    self.magn_latitude  = []
    self.accl_max = 1.0
    self.magn_max = 1.0

  def __len__(self):
    return len(self.raw)

class CalibratedData:
  def __init__(self):
    self.accl = np.array([])
    self.gyro = np.array([])
    self.magn = np.array([])
    self.accl_longitude = np.array([])
    self.accl_latitude  = np.array([])
    self.magn_longitude = np.array([])
    self.magn_latitude  = np.array([])
    self.variance_signal = np.array([])
    self.gravity_samples = np.array([])
    self.all_static_intervals = []
    self.optimal_static_intervals = []

    self.accl_params = np.array(ap)
    self.gyro_params = np.array(gp)
    self.gyro_bias = np.array([0,0,0])
    self.magn_params = np.array(mp)
    self.accl_residual = np.array([])
    self.max_accl_residual = 0.000000001
    self.magn_residual = np.array([])
    self.max_magn_residual = 0.000000001
    self.gyro_residual = np.array([])
    self.max_gyro_residual = 0.000000001

def load_file(filename, rawdata, caldata):
  with open(filename+".json", "r") as f:
    data = json.load(f)
  rawdict = data["rawdata"]
  for k in rawdata.__dict__:
    rawdata.__dict__[k] = rawdict[k]

def save_file(savename, rawdata, caldata):
  with open(savename+".json", "w") as f:
    json.dump({"rawdata":rawdata.__dict__}, f)

def vecToCoords(x, y, z):
  # works with np.arrays as well as floats!
  latitude = np.arctan2(z,np.sqrt(x*x + y*y))
  longitude = np.arctan2(y,x)
  return latitude, longitude
