
from KalmanFilter.genieKFFunction import genieKFTesting
from KalmanFilter.KFFunction import KFTesting

from utilities import matSave 

import hdf5storage as hdf5

test = hdf5.loadmat('data/GETestData94.mat')

data = test['LSandKFTestData'][0]
coeffs = test['testDataInfo'][0]['channelCoefficients']

#print(KFTesting(data, [0.5, -0.4])[1])
#print(KFTesting(data, [0.3, 0.1])[1])
#print(KFTesting(data, [1.949, -0.95])[1])

# predMSEVal = genieKFTesting(data, coeffs)[1]

genieRun = genieKFTesting(data, coeffs, debug=True)

predMSEVal = genieRun[1]

print("Predicted MSE of Genie KF in mode switching: ", predMSEVal)

genKFInstaErrs = genieRun[2]

gkfInfo = dict()

gkfInfo[u'genKFInstaErrs'] = genKFInstaErrs

matSave('logs', 'log', gkfInfo)

# Testing the genie KF performance on the good state dataset
data = test['LSandKFTestData'][1]
coeffs = test['testDataInfo'][1]['channelCoefficients']

#print(KFTesting(data, [0.5, -0.4])[1])
#print(KFTesting(data, [0.3, 0.1])[1])
#print(KFTesting(data, [1.949, -0.95])[1])

# predMSEVal = genieKFTesting(data, coeffs)[1]

genieRun = genieKFTesting(data, coeffs, debug=True)

predMSEVal = genieRun[1]

print("Predicted MSE of Genie KF in only Good States: ", predMSEVal)

genKFInstaErrs = genieRun[2]

gkfInfo = dict()

gkfInfo[u'genKFInstaErrs'] = genKFInstaErrs


# Testing the genie KF performance on the bad state dataset
data = test['LSandKFTestData'][2]
coeffs = test['testDataInfo'][2]['channelCoefficients']

#print(KFTesting(data, [0.5, -0.4])[1])
#print(KFTesting(data, [0.3, 0.1])[1])
#print(KFTesting(data, [1.949, -0.95])[1])

# predMSEVal = genieKFTesting(data, coeffs)[1]

genieRun = genieKFTesting(data, coeffs, debug=True)

predMSEVal = genieRun[1]

print("Predicted MSE of Genie KF in only Bad States: ", predMSEVal)

genKFInstaErrs = genieRun[2]

gkfInfo = dict()

gkfInfo[u'genKFInstaErrs'] = genKFInstaErrs
