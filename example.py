from datagen.GaussMarkovProcess import GaussMarkovSample, GaussMarkovProcess
from datagen.utilities import toeplitzifyData
from datagen.GilbertElliotProcess import gilbertElliotProcess


# test = GaussMarkovProcess(simuLen=10, ar_coefficients=(1.949, -0.95))

# test1 = toeplitzifyData(test[0], seqLen=2)

test2 = gilbertElliotProcess(simuLen=100, transProb=0.1)

print('hello world')