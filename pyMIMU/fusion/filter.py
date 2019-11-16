import numpy as np
from pyquaternion import Quaternion
from time import monotonic
from .triad import TRIAD

class FusionFilter:
  def __init__(self, v_a=None, v_m=None):
    self.q     = Quaternion()
    self.q_dot = Quaternion()
    self.b_hat      = np.array([0.0, 0.0, 0.0])
    self.h          = np.array([0.0, 0.0, 0.0])
    self.b          = np.array([0.0, 0.0, 0.0])
    self.v_a_zero_g = np.array([0.0, 0.0, 0.0])
    self.v_hat_m    = np.array([0.0, 0.0, 0.0])
    self.v_hat_a    = np.array([0.0, 0.0, 0.0])
    self.w_mes      = np.array([0.0, 0.0, 0.0])
    self.now = monotonic()
    self.before = self.now
    self.period = 0
    if v_a is not None and v_m is not None:
      self.initialize_from(v_a, v_m)

  def initialize_from(self, v_a, v_m):
    orientation = TRIAD(
        np.array([0.0,0.0,1.0]), 
        np.array([0.0,1.0,0.0]), 
        v_a, 
        v_m)
    self.q = Quaternion(matrix=orientation).unit

  def calculate_period(self):
    self.now = monotonic()
    self.period = self.now - self.before
    self.before = self.now

  def fuse(self, omega, v_a, v_m, k_I = 1.0, k_P = 3.0, k_a=1.0, k_m=1.0):
    self.calculate_period()
    rotation = q.rotation_matrix
    inverse  = q.conjugate.rotation_matrix

    self.v_a_zero_g = v_a - inverse[:,2]             # zero-g in sensor frame
    self.v_a_zero_g = np.matmul(inverse, v_a_zero_g) # zero-g in global frame

    v_a = v_a.unit
    v_m = v_m.unit
    v_m = np.cross(v_a, v_m)

    self.h = np.matmul(rotation, v_m)
    self.b = np.array([np.sqrt(h[0]*h[0] + h[1]*h[1]), 0, h[2]]);

    self.v_hat_m = np.matmul(inverse, b) # estimated direction of magnetic field
    self.v_hat_a = inverse[:,2] # estimated gravity vector

    # see eqs. (32c) and (48a) in Mahoney et al. 2008
    self.w_mes = ( np.cross(v_a, self.v_hat_a) * k_a 
                 + np.cross(v_m, self.v_hat_m) * k_m )

    # error correction is added to omega (angular rate) before integrating
    if k_I > 0.0:
      self.b_hat += self.w_mes * self.period # see eq. (48c)
      omega += k_P * self.w_mes + k_I * self.b_hat
    else:
      b_hat = np.array([0.0,0.0,0.0]) # Madgwick: "prevent integral windup"
      omega += k_P * w_mes;

    q.integrate(omega, self.period)
    return q
