import argparse as ap
from KalmanFilter.genieKFFunction import genieKFTesting2
from utilities import matSave
import hdf5storage as hdf5

parser = ap.ArgumentParser(description='Script to test performance of Genie Kalman Filter')

parser.add_argument('--testFile', type=str, default='None',
                    help='file containing test data to check performance on'
                    '(default: None)')
parser.add_argument('--initTest', action='store_true', 
                    help="Whether to perform init test functionality"
                    "(default: False)")


args = parser.parse_args()

testFile = args.testFile

testData = hdf5.loadmat(testFile)

testDataLen = len(testData['LSandKFTestData'])

gkfSaveData = dict()
gkfRuns = []

for i in range(0, testDataLen):
    if i == 0:
        # Loading data from the mode switching problem
        data = testData['LSandKFTestData'][i]
        coeffs = testData['testDataInfo'][i]['channelCoefficients']

        genieRun = genieKFTesting2(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in mode switching: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs
        
        gkfRuns.append(gkfRunInfo)
    
    if i == 1:
        # Testing the genie KF performance on the good state dataset
        data = testData['LSandKFTestData'][i]
        coeffs = testData['testDataInfo'][i]['channelCoefficients']

        genieRun = genieKFTesting2(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in only Good States: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs

        gkfRuns.append(gkfRunInfo)

    if i == 2:
        # Testing the genie KF performance on the bad state dataset
        data = testData['LSandKFTestData'][2]
        coeffs = testData['testDataInfo'][2]['channelCoefficients']

        genieRun = genieKFTesting2(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in only Bad States: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs

        gkfRuns.append(gkfRunInfo)

gkfSaveData['gkfRuns'] = gkfRuns
gkfSaveData['file'] = testFile
gkfSaveData['initTest'] = args.initTest

matSave('logs', 'log', gkfSaveData)
