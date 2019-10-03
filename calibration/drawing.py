import numpy as np
import asyncio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

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

class Artist:
  def __init__(self, fig, ax, xlim, ylim, fmt, alpha, updatex, updatey):
    self.fig = fig
    self.ax = ax
    if xlim is None:
      self.ax.set_xlim(auto=True)
    else:
      self.ax.set_xlim(*xlim)
    if xlim is None:
      self.ax.set_ylim(auto=True)
    else:
      self.ax.set_ylim(*ylim)
    self.fmt = fmt
    self.alpha = alpha
    self.updatex = updatex
    self.updatey = updatey

  def setup(self):
    artists = self.ax.plot(
        self.updatex(), 
        self.updatey(), 
        self.fmt, 
        alpha=self.alpha)
    self.artist = artists[-1]
    self.artist.set_alpha(self.alpha)

  def draw(self):
    self.artist.set_data(self.updatex(), self.updatey())
    return self.artist

class AsyncAnimationEventSource:
  def __init__(self):
    self._callbacks = []
    self.interval = 0 # I actually just don't care
    self.started = False

  def add_callback(self, cb):
    self._callbacks.append(cb)

  def remove_callback(self, cb):
    for _cb in self._callbacks:
      if _cb is cb: 
        self.callbacks.remove(_cb)
    return

  def start(self):
    self.started = True

  def stop(self):
    self.started = False

  def tick(self):
    for cb in self._callbacks:
      cb()


async def draw_loop(continue_flag, rawdata, caldata):
  print("Setting up gui")
  fig, axes = plt.subplots(2,5)
  artists = get_artists(fig, axes, rawdata, caldata)
  for a in artists: a.setup()
  for row in axes: 
    for ax in row:
      ax.set_xticks([])
      ax.set_yticks([])
  
  frame_ticker = AsyncAnimationEventSource()
  aniation = FuncAnimation(fig, 
      (lambda frame: [a.draw() for a in artists]), 
      count(), event_source=frame_ticker, blit=True)

  plt.show(False)
  plt.draw()
  plt.pause(0.001)

  while continue_flag:
    if len(rawdata) <= 0:
      await asyncio.sleep(0.06)
      continue
    frame_ticker.tick() # generates an event which causes the animation to tick
    fig.canvas.start_event_loop(0.001)

    await asyncio.sleep(0.06)

  print("Closing gui")
  plt.close()
  return

class BgCache:
  def __init__(self, cache):
    self._cache = cache

  def get(self):
    return self._cache

  def set(self, newcache):
    self._cache = newcache

