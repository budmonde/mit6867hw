class SquaredLossSampler:
  def __init__(self, X, y):
    self.X = X
    self.y = y
  def sample(self, theta):
    loss = np.dot(self.X, theta)- self.y
    return np.sum(loss ** 2)* 0.5
  def sampleGrad(self, theta):
    loss = np.dot(self.X, theta) - self.y
    return np.dot(self.X.T, loss)
