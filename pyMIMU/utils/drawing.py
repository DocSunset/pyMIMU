import numpy as np
import asyncio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count

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

async def draw_loop(continue_flag, subplots, get_artists_func, *get_artists_args):
  # supply a function which returns a list of artists. Their draw() methods will
  # be called at each animation frame allowing you to update their data
  print("Setting up gui")
  fig, axes = plt.subplots(*subplots)
  artists = get_artists_func(fig, axes, *get_artists_args)
  for a in artists: a.setup()
  for row in axes: 
    for ax in row:
      ax.set_xticks([])
      ax.set_yticks([])
  
  frame_ticker = AsyncAnimationEventSource()
  aniation = FuncAnimation(fig, 
      (lambda frame: [a.draw() for a in artists]), 
      count(), event_source=frame_ticker, blit=True)

  plt.show(block=False)
  plt.draw()
  plt.pause(0.001)

  while continue_flag:
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
