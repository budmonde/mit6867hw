import numpy as np
from act_loss_functions import *
from read_data import *

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
    elif init == "Ones":
      self.w = np.ones((width, bottom+1))

  def check(self):
    assert self.state.size == self.width
    assert self.dstate.size == self.width
    assert self.bottom_out.shape == (self.bottom+1, 1)
    assert self.w.shape == (self.width, self.bottom+1)
    #print self

  def forward(self, bottom_out):
    bottom_out = bottom_out.reshape((bottom_out.ravel().size, 1))

    assert bottom_out.shape == (self.bottom, 1)

    self.bottom_out = np.insert(bottom_out, 0, 1).reshape((self.bottom+1, 1))
    self.state = self.activation.output(self.w.dot(self.bottom_out))

#    if self.activation.type == "Softmax":
#      print "here"
#      print self.state
#      print "done"
#
    self.check()

    return self.state

  def backprop(self, top_w, top_back):
    assert top_w.shape == (self.top, self.width+1)
    assert top_back.size == self.top

    der = self.activation.derivative(self.state).ravel()
    print "#######"
    print der
    print "#######"
    self.dstate = top_back.dot(top_w)[1:] * der

    self.check()

    return self.dstate

  def update(self):
    self.w -= self.lrate * self.bottom_out.T * self.dstate.reshape((self.width,1))

    self.check()

  def __str__(self):
    return "bottom:\t%s\nwidth:\t%s\ntop:\t%s\nactivation:\t%s\nstate:\t%s\ndstate\t%s\nw:\n%s\n" % (self.bottom, self.width, self.top, self.activation, self.state.ravel(), self.dstate, self.w)

class NeuralNet:

  def __init__(self, layers, activation, out_activation, loss, lrate, init=None):
    # functions
    self.loss = loss
    self.out_activation = out_activation

    # constants
    self.lrate = lrate
    self.data_sz = layers[0]
    self.label_sz = layers[-1]

    # setting up layers
    top_sz = self.data_sz
    self.layers = []
    for i in range(1, len(layers)-1):
      self.layers.append(Layer(top_sz, layers[i], layers[i+1], activation, lrate, init=init))
      top_sz = layers[i]
    self.layers.append(Layer(top_sz, layers[-1], 1, out_activation, lrate, init=init))

    # state variables
    self.state = 0.0

  def check(self):
    assert type(self.state) is float
    for i in self.layers:
      i.check()

  def forward(self, data, label):
    top_out = data
    for i in range(len(self.layers)):
      top_out = self.layers[i].forward(top_out)
    self.state = float(self.loss((label, self.layers[-1].state)))

    self.check()

    return self.layers[-1].state

  def backprop(self):
    top_back = np.ones(1)
    top_w = np.ones(self.label_sz+1).reshape(1,self.label_sz+1)
    for i in range(1, len(self.layers)+1):
      top_back = self.layers[-i].backprop(top_w, top_back)
      top_w = self.layers[-i].w

    self.check()

  def update(self):
    for i in self.layers:
      i.update()

    self.check()

  def cycle(self, data, label):
    self.forward(data, label)
    self.backprop()
    self.update()

  def SGD(self, data, labels, threshold):
    diff = np.inf
    err = np.inf
    while diff > threshold:
      for i in range(data.shape[0]):
        self.cycle(data[i,:], labels[i,:])
        print self.state
      diff = abs(err - self.state)
      err = self.state
      
  def __str__(self):
    string = ""
    for i in self.layers:
      string += "LAYER:\n"
      string += str(i)
    string += "loss:\t"
    string += str(self.state)
    return string
      
    
if __name__ == "__main__":
  activation = Function(ReLU, ReLU_derivative, "ReLU")
  out_activation = Function(softmax, delta_L, "Softmax")
  loss = cross_entropy
  net = NeuralNet([2,5,3], activation, out_activation, loss, 0.1, init="Ones")
  data, labels = read_data()
  i=0
  print "\n\n###INPUT###\n\n"
  print data[i,:], labels[i,:]
  print "\n\n###INIT###\n\n"
  print net
  print "\n\n###FORWARD###\n\n"
  net.forward(data[i,:], labels[i,:])
  print net
  print "\n\n###BACKPROP###\n\n"
  net.backprop()
  print net
  print "\n\n###UPDATE###\n\n"
  net.update()
  print net

#  net.SGD(data, labels, 0.1)
#  i = 31
#  print "LABEL:"
#  print data[i,:], labels[i,:]
#  print "PREDICTION"
#  print net.forward(data[i,:], labels[i,:])

  ## Testing Layer
  #layer = Layer(3, 2, 4, activation, 0.1, init=None)
  #layer.w = np.ones((layer.width, layer.bottom+1))
  #print "FORWARD:"
  #print layer.forward(np.asarray([2,3,4]))
  #print "BACKWARD:"
  #print layer.backprop(np.arange(1,13).reshape((4,3)), np.asarray([1,2,3,4]))
  #print "UPDATE:"
  #layer.update()
