
from KalmanFilter.genieKFFunction import genieKFTesting
from KalmanFilter.KFFunction import KFTesting

import hdf5storage as hdf5

test = hdf5.loadmat('data/GETestData2.mat')

data = test['LSandKFTestData'][0]
coeffs = test['testDataInfo'][0]

print(KFTesting(data, [0.3, 0.1])[1])
print(KFTesting(data, [1.949, -0.95])[1])

predMSEVal = genieKFTesting(data, coeffs)[1]

print(predMSEVal)
