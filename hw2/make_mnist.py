import numpy as np
import csv
import os

TRAIN_SZ = 200
VAL_SZ = 150
TEST_SZ = 150
TOTAL_SZ = TRAIN_SZ + VAL_SZ + TEST_SZ
VECTOR_SZ = 784

def makeDataset(plist, nlist, normalize=False):
  pdigits = len(plist)
  ndigits = len(nlist)
  totlist = plist + nlist

  total_sz = pdigits * TOTAL_SZ + ndigits * TOTAL_SZ
  p_sz = pdigits * TOTAL_SZ

  X = np.zeros((total_sz, VECTOR_SZ))

  k = 0
  for filename in totlist:
    with open(filename, 'r') as f:
      reader = csv.reader(f, delimiter=" ")
      X[k*TOTAL_SZ:(k+1)*TOTAL_SZ] = np.asarray([[int(s) for s in row] for i, row in enumerate(reader) if i < TOTAL_SZ])
    k += 1

  if normalize:
    X = 2*X/255. - 1

  Y = np.ones((total_sz))
  Y[p_sz:] *= -1
  Y = Y[:, np.newaxis]

  pos_tr = 0
  pos_v = pos_tr + pdigits*(TRAIN_SZ)
  pos_t = pos_v + pdigits*(VAL_SZ)
  neg_tr = p_sz
  neg_v = neg_tr + ndigits*(TRAIN_SZ)
  neg_t = neg_v + ndigits*(VAL_SZ)

  train_X = np.concatenate((X[pos_tr:pos_v,:], X[neg_tr:neg_v,:]), axis=0)
  val_X = np.concatenate((X[pos_v:pos_t,:], X[neg_v:neg_t,:]), axis=0)
  test_X = np.concatenate((X[pos_t:neg_tr,:], X[neg_t:,:]), axis=0)

  train_Y = np.concatenate((Y[pos_tr:pos_v], Y[neg_tr:neg_v]))
  val_Y = np.concatenate((Y[pos_v:pos_t], Y[neg_v:neg_t]))
  test_Y = np.concatenate((Y[pos_t:neg_tr], Y[neg_t:]))

  return train_X, val_X, test_X, train_Y, val_Y, test_Y

