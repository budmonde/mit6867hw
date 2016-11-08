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

  def check(self):
    assert self.state.size == self.width
    assert self.dstate.size == self.width
    assert self.bottom_out.shape == (self.bottom+1, 1)
    assert self.w.shape == (self.width, self.bottom+1)

  def forward(self, bottom_out):
    bottom_out = bottom_out.reshape((bottom_out.ravel().size, 1))
    assert bottom_out.shape == (self.bottom, 1)

    self.bottom_out = np.insert(bottom_out, 0, 1).reshape((self.bottom+1, 1))
    self.state = activation.output(self.w.dot(self.bottom_out))

    return self.state

  def backprop(self, top_w, top_back):
    assert top_w.shape == (self.top, self.width+1)
    assert top_back.size == self.top

    der = activation.derivative(self.state).ravel()
    self.dstate = top_back.dot(top_w)[1:] * der

    return self.dstate

  def update(self):

    self.w -= self.lrate * self.bottom_out.T * self.dstate.reshape((self.width,1))

  def __str__(self):
    return self.bottom, self.width, self.top, self.state, self.dstate

class NeuralNet:

  def __init__(self, data_sz, label_sz, layers, activation, class_func, loss, lrate):
    # functions
    self.loss = loss
    self.class_func = class_func

    # constants
    self.lrate = lrate
    self.data_sz = data_sz
    self.label_sz = label_sz

    # setting up layers
    top_sz = self.inp_sz
    self.layers = []
    for i in range(layers-1):
      self.layers.append(Layer(top_sz, layers[i], layers[i+1], activation, lrate, init=None))
      top_sz = layers[i]
    self.layers.append(Layer(top_sz, layers[-1], label_sz, activation, lrate, init=None))

    # state variables
    self.prob = np.ones(label_sz) / label_sz
    self.state = 0
    self.dstate = np.zeros((label_sz))

  def check(self):
    assert self.prob.size == self.label_sz
    assert type(self.state) is int
    assert self.dstate.size == self.label_sz

    for i in layers:
      layers.check()

  def forward(self, data, label):
    top_out = data
    for i in range(len(self.layers)):
      top_out = self.layers[i].forward(top_out)
    self.prob = self.class_func(top_out)
    self.state = self.loss.function(top_out)

    self.check()

    return self.prob

  def backprop(self):
    self.dstate = self.loss.function(self.state)
    top_back = self.dstate
    top_w = np.ones(label_sz)
    for i in range(1, len(self.layers)+1):
      top_back = self.layers[i].backprop(top_w, top_back)
      top_w = self.layers[i].w

    self.check()
    
    
    
    
    
if __name__ == "__main__":
  activation = Function(ReLU, ReLU_derivative)
  layer = Layer(3, 2, 4, activation, 0.1, init=None)
  layer.w = np.ones((layer.width, layer.bottom+1))
  print "FORWARD:"
  print layer.forward(np.asarray([2,3,4]))
  print "BACKWARD:"
  print "w", np.arange(1,13).reshape((4,3))
  print layer.backprop(np.arange(1,13).reshape((4,3)), np.asarray([1,2,3,4]))
  print "UPDATE:"
  layer.update()
