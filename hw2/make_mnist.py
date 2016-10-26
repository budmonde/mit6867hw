import numpy as np
import csv
import os

TRAIN_SZ = 200
VAL_SZ = 150
TEST_SZ = 150
TOTAL_SZ = TRAIN_SZ + VAL_SZ + TEST_SZ
VECTOR_SZ = 784

def makeDataset(plist, nlist, normalize=False):
  totlist = plist + nlist
  pdigits = len(plist)
  ndigits = len(nlist)
  digits = len(totlist)
  
  total_sz = digits * TOTAL_SZ
  n_sz = ndigits * TOTAL_SZ
  p_sz = pdigits * TOTAL_SZ

  train_X = np.zeros((digits * TRAIN_SZ, VECTOR_SZ))
  val_X = np.zeros((digits * VAL_SZ, VECTOR_SZ))
  test_X = np.zeros((digits * TEST_SZ, VECTOR_SZ))

  train_Y = np.ones((digits*TRAIN_SZ))
  train_Y[pdigits*TRAIN_SZ:] *= -1
  train_Y = train_Y[:, np.newaxis]

  val_Y = np.ones((digits*VAL_SZ))
  val_Y[pdigits*VAL_SZ:] *= -1
  val_Y = val_Y[:, np.newaxis]

  test_Y = np.ones((digits*TEST_SZ))
  test_Y[pdigits*TEST_SZ:] *= -1
  test_Y = test_Y[:, np.newaxis]

  k = 0
  for filename in totlist:
    with open(filename, 'r') as f:
      reader = csv.reader(f, delimiter=" ")
      temp = np.asarray([[int(s) for s in row] for i, row in enumerate(reader) if i < TOTAL_SZ])
      train_X[k*TRAIN_SZ:(k+1)*TRAIN_SZ] = temp[:TRAIN_SZ]
      val_X[k*VAL_SZ:(k+1)*VAL_SZ] = temp[TRAIN_SZ:TRAIN_SZ+VAL_SZ]
      test_X[k*TEST_SZ:(k+1)*TEST_SZ] = temp[TRAIN_SZ+VAL_SZ:]
    k += 1

  if normalize:
    train_X = 2*train_X/255. - 1
    val_X = 2*val_X/255. - 1
    test_X = 2*test_X/255. - 1

  return train_X, val_X, test_X, train_Y, val_Y, test_Y
