import numpy as np

class BaseEmptyClass:
  def __init__(self):
    print("stub")

class x(BaseEmptyClass):
  def __init__(self):
    super(x, self).__init__()
    self.a = 3
    print("x")

class y(BaseEmptyClass):
  def __init__(self):
    super(y, self).__init__()
    self.b = 5
    print("y")

class z(x,y):
  def __init__(self):
    super(z, self).__init__()
    print(self.a, self.b)

c = z()