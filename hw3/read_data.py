import numpy as np

def read_data(filename):
  dataset = np.loadtxt(filename)
  data = dataset[:,:2]
  labels = dataset[:,2:]
  return data, labels

#if __name__ == "__main__":
#  data, labels = read_data()
#  print "DATA"
#  print data[:5,:]
#  print "shape:"
#  print data.shape
#  print "LABELS"
#  print labels[:5,:]
#  print "shape:"
#  print labels.shape

def read_data_MNIST(start, end, normalize=False):
  size = end - start
  for i in range(10):
    dataset = np.loadtxt("data/mnist_digit_"+str(i)+".csv")
    if i == 0:
      data = dataset[start:end,:]
      labels = np.ones((size,1)) * i
    else:
      data = np.concatenate((data, dataset[start:end,:]))
      labels = np.concatenate((labels, np.ones((size,1)) * i))
  order = np.arange(data.shape[0])
  np.random.shuffle(order)
  data = data[order]
  labels = labels[order]
  if normalize:
    data = (data * 2. / 255) - 1
  return data, labels

if __name__ == "__main__":
  data, labels = read_data_MNIST(0,200,normalize=True)
  print np.median(data)
  print labels
  print data.shape, labels.shape
