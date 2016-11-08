import numpy as np
class Layer:

  def __init__(self, bottom, width, top, activation, init=None):
    self.bottom = bottom
    self.width = width
    self.top = top
    self.state = np.zeros(width+1)
    self.activation = activation
    if init == None:
      self.w = np.zeros((bottom, width+1))
    elif init == "Gauss":
      sigma = 1./(bottom**0.5)
      self.w = np.random.normal(0, sigma, (bottom, width+1))

  def forward(self, bottom_out):
    assert bottom_out.size == self.w.shape[1]
    z = self.w.dot(np.insert(bottom_out, 0, 1))
    self.state = activation.output(z)
    return self.state

  def back(self, top_w, top_out):
    dstate = activation.derivative(self.state)
    assert dstate.size == top_out.size
    return dstate * top_w * top_out

class NeuralNet:

  def __init__(self, layers, activation, loss):
    self.layers = []
