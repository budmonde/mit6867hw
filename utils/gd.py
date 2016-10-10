class GradientDescent:
  def __init__(self, f, df=None):
    self.f = f
    if df is None:
      self.df = lambda X, h: self.estimateGrad(self, X, h)
    else:
      self.df = df

  def estimateGrad(self, X, h):
    grad = []
    X_size = np.size(X)
    for i in xrange(X_size):
      step = np.zeros(X_size)
      step[i] = h
      grad.append((self.f(X + 0.5*step) - self.f(X - 0.5*step)) / h)
    return np.array(grad)

  def run(self, init, epsilon, step=lambda x: 1.0):
    cur_val = np.copy(init)
    grad = self.df(cur_val)
    iteration = 0
    while np.linalg.norm(grad) > epsilon:
        cur_val -= step(iteration) * grad
        grad = self.df(cur_val)
        iteration += 1
    return (cur_val, iteration)

def BGD(X, y, model, init, epsilon):
  sampler = model(X, y)
  f = lambda theta: sampler.sample(theta)
  df = lambda theta: sampler.sampleGrad(theta)
  gd = GradientDescent(f, df)
  return gd.run(init, epsilon)
"""
def SGD(X, y, model, init, epsilon, repeates, step):
  avg_theta = np.zeros(repeats) #this looks sketchy
  avg_iterations = 0
  avg_err = 0

  for repeat in xrange(repeats):
    shuff_i = np.shuffle(np.arange(self.y.size))
    shuff_X = X[shuff_i]
    shuff_y = y[shuff_i]
    X_0 = np.copy(shuff_X[0])
    X_0.shape = (1, 10)
    y_0 = np.array([np.copy(y[0])])
    sgdJ = J(X_0, y_0)
    sgdJGrad = JGrad(X_0, y_0)
    step = learningRate(1e7, 0.9)

    current_value = np.ones(10)
    gradient = sgdJGrad(current_value)
    iteration = 0
    epsilon = 1e-3

    while np.linalg.norm(gradient) > epsilon:
        i = random.randint(0,99)
        X_i = np.copy(X[i])
        X_i.shape = (1,10)
        y_i = np.array([np.copy(y[i])])
        sgdJ = J(X_i, y_i)
        sgdJGrad = JGrad(X_i, y_i)
        current_value -= step(iteration) * gradient
        gradient = sgdJGrad(current_value)
        iteration += 1
    average_theta += current_value
    average_iterations += iteration
    average_error += errorFunction(current_value)
  return (average_theta, iteration, average_error)
"""




"""
    Given a and b, returns a function that, given the iteration
    t, returns the learning rate corresponding to that iteration
"""
def learningRate(a, b):
    assert a >= 0
    assert b > 0.5
    assert b < 1.0
    def learningRateSampler(t):
        return (a + t) ** -b
    return learningRateSampler

