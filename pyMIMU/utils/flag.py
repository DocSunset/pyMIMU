class Flag:
  def __init__(self):
    self._flag = False

  def set(self, b):
    self._flag = bool(b)

  def __bool__(self):
    return self._flag

  def __nonzero__(self):
    return self._flag
