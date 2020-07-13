
from KalmanFilter.genieKFFunction import genieKFTesting
from KalmanFilter.KFFunction import KFTesting

from utilities import matSave 

import hdf5storage as hdf5

test = hdf5.loadmat('data/GETestData3.mat')

data = test['LSandKFTestData'][0]
coeffs = test['testDataInfo'][0]['channelCoefficients']

#print(KFTesting(data, [0.5, -0.4])[1])
#print(KFTesting(data, [0.3, 0.1])[1])
#print(KFTesting(data, [1.949, -0.95])[1])

# predMSEVal = genieKFTesting(data, coeffs)[1]

genieRun = genieKFTesting(data, coeffs, debug=True)

predMSEVal = genieRun[1]

print(predMSEVal)

genKFInstaErrs = genieRun[2]

gkfInfo = dict()

gkfInfo[u'genKFInstaErrs'] = genKFInstaErrs

matSave('logs', 'log', gkfInfo)
