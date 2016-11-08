import numpy as np
from act_loss_functions import *
class Layer:

  def __init__(self, bottom, width, top, activation, lrate, init=None):
    # constants
    self.bottom = bottom
    self.width = width
    self.top = top
    self.activation = activation
    self.lrate = lrate

    # state variables
    self.state = np.zeros(width)
    self.dstate = np.zeros(width)
    self.bottom_out = np.zeros(bottom+1).reshape((bottom+1, 1))

    # weights
    if init == None:
      self.w = np.zeros((width, bottom+1))
    elif init == "Gauss":
      sigma = 1./(bottom**0.5)
      self.w = np.random.normal(0, sigma, (width, bottom+1))

  def forward(self, bottom_out):
    bottom_out = bottom_out.reshape((bottom_out.ravel().size, 1))
    assert bottom_out.shape[0] == self.bottom and bottom_out.shape[1] == 1

    self.bottom_out = np.insert(bottom_out, 0, 1).reshape((self.bottom+1, 1))
    self.state = activation.output(self.w.dot(self.bottom_out))

    assert self.bottom_out.shape[0] == self.bottom+1 and self.bottom_out.shape[1] == 1
    assert self.state.size == self.width

    return self.state

  def back(self, top_w, top_back):
    assert top_w.shape[0] == self.top and top_w.shape[1] == self.width+1
    assert top_back.size == self.top

    der = activation.derivative(self.state).ravel()
    self.dstate = top_back.dot(top_w)[1:] * der

    assert self.dstate.size == self.width

    return self.dstate

  def update(self):

    self.w -= self.lrate * self.bottom_out.T * self.dstate.reshape((self.width,1))

class NeuralNet:

  def __init__(self, layers, activation, loss):
    self.layers = []



if __name__ == "__main__":
  activation = Function(ReLU, ReLU_derivative)
  layer = Layer(3, 2, 4, activation, 0.1, init=None)
  layer.w = np.ones((layer.width, layer.bottom+1))
  print "FORWARD:"
  print layer.forward(np.asarray([2,3,4]))
  print "BACKWARD:"
  print "w", np.arange(1,13).reshape((4,3))
  print layer.back(np.arange(1,13).reshape((4,3)), np.asarray([1,2,3,4]))
  print "UPDATE:"
  layer.update()
