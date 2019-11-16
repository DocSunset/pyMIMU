def TRIAD(v1, v2, w1, w2):
  for v in (v1, v2, w1, w2):
    v = v / np.linalg.norm(v)
  r1 = v1;
  r2 = np.cross(r1, v2)
  r3 = np.cross(r1, r2)
  m1 = np.hstack(r1.T, r2.T, r3.T)

  s1 = w1;
  s2 = np.cross(s1, w2)
  s3 = np.cross(s1, s2)
  m2 = np.hstack(s1.T, s2.T, s3.T)

  return m2 @ m1.T

