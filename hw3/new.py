import numpy as np
from matplotlib import pyplot as plt
from act_loss_functions import *
from read_data import *

class Weight:

  def __init__(self, bottom, top, lrate, init):
    self.bottom = bottom
    self.top = top
    self.lrate = lrate
    if init == None:
      self.w = np.zeros((bottom, top))
    elif init == "Gauss":
      sigma = 1./(bottom**0.5)
      self.w = np.random.normal(0, sigma, (bottom, top))
    elif init == "Ones":
      self.w = np.ones((bottom, top))

  def check(self):
    assert self.w.shape == (self.bottom, self.top)

  def update(self, bottom_out, top_back):
    assert bottom_out.size == self.bottom
    assert top_back.size == self.top
    bottom_out = bottom_out.reshape((bottom_out.size, 1))
    top_back = top_back.reshape((1, top_back.size))
    self.w -= self.lrate * bottom_out * top_back

    self.check()

class InputLayer:

  def __init__(self, width, topW):
    assert topW.bottom == width
    self.width = width
    self.topW = topW

    self.out = np.zeros(width)

  def forward(self, data):
    assert data.size == self.width
    self.out = data
    return self.out

  def backprop(self):
    return

class FCLayer:

  def __init__(self, botW, width, topW, activation, lrate):
    assert botW.top == width
    assert topW.bottom == width
    self.botW = botW
    self.width = width
    self.topW = topW
    self.activation = activation
    self.lrate = lrate

    self.bias = np.zeros(width)

    self.out = np.zeros(width)
    self.der = np.zeros(width)
    self.back = np.zeros(width)

  def check(self):
    assert self.bias.size == self.width
    assert self.out.size == self.width
    assert self.der.size == self.width
    assert self.back.size == self.width


  def forward(self, bottom_out):
    assert bottom_out.size == self.botW.bottom
    z = bottom_out.dot(self.botW.w) + self.bias
    self.out = self.activation.output(z)
    self.der = self.activation.derivative(z)

    self.check()

    return self.out

  def backprop(self, top_back):
    assert top_back.size == self.topW.top
    self.back = self.der * top_back.dot(self.topW.w.T)
    self.bias -= self.lrate * self.back
    
    self.check()

    return self.back

class OutputLayer:

  def __init__(self, botW, width, activation, lrate):
    assert botW.top == width
    self.botW = botW
    self.width = width
    self.activation = activation
    self.lrate = lrate

    self.bias = np.zeros(width)

    self.out = np.zeros(width)
    self.back = np.zeros(width)

  def check(self):
    assert self.bias.size == self.width
    assert self.out.size == self.width
    assert self.back.size == self.width


  def forward(self, bottom_out, label):
    assert bottom_out.size == self.botW.bottom
    z = bottom_out.dot(self.botW.w) + self.bias
    self.out = self.activation.output(z)
    self.back = self.activation.derivative((label, self.out))

    self.check()

    return self.out

  def backprop(self):
    self.bias -= self.lrate * self.back
    
    self.check()

    return self.back

class NeuralNet:

  def __init__(self, layers, activation, output_activation, loss_fn, lrate, init=None):
    # Constants
    self.loss_fn = loss_fn

    # Weights
    self.weights = []
    for i in xrange(len(layers)-1):
      self.weights.append(Weight(layers[i], layers[i+1], lrate, init))
      
    # Layers
    self.layers = []
    self.layers.append(InputLayer(layers[0], self.weights[0]))
    for i in xrange(1, len(layers)-1):
      self.layers.append(FCLayer(self.weights[i-1], layers[i], self.weights[i], activation, lrate))
    self.layers.append(OutputLayer(self.weights[-1], layers[-1], output_activation, lrate))

  def push(self, data, label):
    output = self.layers[0].forward(data)
    for i in xrange(1, len(self.layers)-1):
      output = self.layers[i].forward(output)
    output = self.layers[-1].forward(output, label)
    return output, float(self.loss_fn((label, output)))

  def pull(self):
    back = self.layers[-1].backprop()
    for i in xrange(len(self.layers)-2, 0, -1):
      back = self.layers[i].backprop(back)
    back = self.layers[0].backprop()

    for i in xrange(len(self.weights)):
      self.weights[i].update(self.layers[i].out, self.layers[i+1].back)

  def stats(self, data, labels):
    loss, hits = 0.0, 0.0
    for i in xrange(data.shape[0]):
      loss += self.push(data[i,:], labels[i,:])[1]
      hits += (labels[i,:] == np.argmax(self.push(data[i,:], labels[i,:])[0]))
    return float(loss/data.shape[0]), float(hits/data.shape[0])

  def SGD(self, train_X, train_Y, val_X, val_Y, threshold, max_it):
    assert train_X.shape[1] == self.layers[0].width
    assert val_X.shape[1] == self.layers[0].width
    it = 0
    diff = np.inf
    prev_loss = np.inf
    while diff > threshold and it < max_it:
      if it % 50 == 0:
        print it
      it += 1
      for i in xrange(train_X.shape[0]):
        self.push(train_X[i,:], train_Y[i,:])[1]
        self.pull()
      loss, acc = self.stats(val_X, val_Y)
      diff = abs(prev_loss - loss)
      prev_loss = loss
    if it > max_it:
      print "Reached max_it"

def plot_network(net, X_data, Y_data, title):
  colors = [['ro','rx'], ['bo','bx'], ['go','gx'], ['co','cx'], ['mo','mx'], ['yo','yx'], ['ko','kx']]

  num_classes = net.layers[0].width
  accuracy = 0

  classifications = {}
  for i in xrange(num_classes):
    classifications[i] = {}
    for j in xrange(2):
      classifications[i][j] = []

  for i in xrange(len(X_data)):
    cur_x = X_data[i]
    cur_y = Y_data[i]

    output_value = net.push(cur_x, cur_y)[0]
    classification = np.argmax(output_value)
    if classification == int(cur_y):
      classifications[classification][0].append(cur_x)
      accuracy += 1
    else:
      classifications[classification][1].append(cur_x)

  print "Accuracy: ", accuracy/float(len(X_data))
  for i in xrange(num_classes):
    for j in xrange(2):
      xs = [thing[0] for thing in classifications[i][j]]
      ys = [thing[1] for thing in classifications[i][j]]

      color = colors[i][j]

      plt.plot(xs, ys, color)
  plt.title(title)
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.tight_layout()
  plt.show() 

if __name__ == "__main__":
  ### Plots formatting ###

  font = {'family':'serif',
          'weight':'normal',
          'size': 20}
  plt.rc('font', **font)

  # MNIST dataset
  print "LOADING"
  train_X, train_Y = read_data_MNIST(0,200,normalize=True)
  val_X, val_Y = read_data_MNIST(200,400,normalize=True)
  test_X, test_Y = read_data_MNIST(200,400,normalize=True)

#  # dataset
#  number = str(4)
#  title = "Dataset" + number
#  train_X, train_Y = read_data("data/data"+number+"_train.csv")
#  val_X, val_Y = read_data("data/data"+number+"_validate.csv")
#  test_X, test_Y = read_data("data/data"+number+"_test.csv")
#  train_Y[train_Y<0] = 0
#  val_Y[val_Y<0] = 0
#  test_Y[test_Y<0] = 0

  # functions
  activation = Function(ReLU, ReLU_derivative, "ReLU")
  output_activation = Function(softmax, delta_L2, "Softmax")
  loss_fn = cross_entropy

  # constants
  threshold = 0.0001
  init = "Gauss"
  max_it = 500

  # no HU optimize routine
#  lrate_list = [0.001, 0.1]
#  network = [784,10]
#  for lrate in lrate_list:
#    net = NeuralNet(network, activation, output_activation, loss_fn, lrate, init)
#    # run
#    print "RUNNING\tlrate=" + str(lrate) + "\tnetwork" + str(network)
#    net.SGD(train_X, train_Y, val_X, val_Y, threshold, max_it)
#    print "\t\t\t\t\tTRAIN\t\t\t\t", net.stats(train_X, train_Y)
#    print "\t\t\t\t\tVALIDATION\t\t\t", net.stats(val_X, val_Y)
#    print "\t\t\t\t\tTEST\t\t\t\t", net.stats(test_X, test_Y)

  # one HU optimize routine
#  lrate_list = [0.0001, 0.001]
#  hid_width = [10,100,500]
#  network = [784,10,10]
#  for i in hid_width:
#    network[1] = i
#    for lrate in lrate_list:
#      net = NeuralNet(network, activation, output_activation, loss_fn, lrate, init)
#      # run
#      print "RUNNING\tlrate=" + str(lrate) + "\tnetwork" + str(network)
#      net.SGD(train_X, train_Y, val_X, val_Y, threshold, max_it)
#      print "\t\t\t\t\tTRAIN\t\t\t\t", net.stats(train_X, train_Y)
#      print "\t\t\t\t\tVALIDATION\t\t\t", net.stats(val_X, val_Y)
#      print "\t\t\t\t\tTEST\t\t\t\t", net.stats(test_X, test_Y)

  # two HU optimze routine
  lrate_list = [0.0001, 0.001]
  hid_width = [500]
  network = [784,10,10,10]
  for i in hid_width:
    network[1] = i
    for j in hid_width:
      network[2] = j
      for lrate in lrate_list:
        net = NeuralNet(network, activation, output_activation, loss_fn, lrate, init)
        # run
        print "RUNNING\tlrate=" + str(lrate) + "\tnetwork" + str(network)
        net.SGD(train_X, train_Y, val_X, val_Y, threshold, max_it)
        print "\t\t\t\t\tTRAIN\t\t\t\t", net.stats(train_X, train_Y)
        print "\t\t\t\t\tVALIDATION\t\t\t", net.stats(val_X, val_Y)
        print "\t\t\t\t\tTEST\t\t\t\t", net.stats(test_X, test_Y)

#  # plot routine:
#  lrate = 0.01
#  network = [2,5,2]
#  net = NeuralNet(network, activation, output_activation, loss_fn, lrate, init)
#  net.SGD(train_X, train_Y, val_X, val_Y, threshold, max_it)
#  plot_network(net, test_X, test_Y, title)
  
