from datagen.GaussMarkovProcess import GaussMarkovSample, GaussMarkovProcess
from datagen.utilities import toeplitzifyData
from datagen.GilbertElliotProcess import gilbertElliotProcess
from LeastSquares.LeastSquares import fitLSFilter

import numpy as np


[x, z, _, _] = GaussMarkovProcess(simuLen=20, ar_coeffs=(0.3, 0.1))

obsMat = toeplitzifyData(z, seqLen=10)[:, 0:-1]
procMat = toeplitzifyData(x, seqLen=10)
predProcVals = procMat[:, -1]
estProcVals = procMat[:, -2]

coefs1 = fitLSFilter(obsMat, predProcVals)

# Verifying the functionality of the LS coeffs
[x, z, _, _] = GaussMarkovProcess(simuLen=20, ar_coeffs=(0.3, 0.1), seed=1)

obsMat = toeplitzifyData(z, seqLen=10)[:, 0:-1]
procMat = toeplitzifyData(x, seqLen=10)
predProcVals = procMat[:, -1]
estProcVals = procMat[:, -2]

# TODO: Figure out why this number is not converging to what we expect it to converge to
l1 = np.mean(np.square(np.abs(np.matmul(obsMat, coefs1) - predProcVals)))


# test2 = gilbertElliotProcess(simuLen=100, transProb=0.1)

print('hello world')