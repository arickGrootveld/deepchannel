from mismatch_data_gen import ARDatagenMismatch
from utilities import convertToBatched, matSave
import hdf5storage as hdf5s
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Generating Test Data from a linear set of stable AR Coefficients')

parser.add_argument('--testLen', type=float, default=1e2,
                    help='length of the test data to be generated')

parser.add_argument('--seqLen', type=float, default=15,
                    help='length of the sequences of data to be generated')

parser.add_argument('--seed', type=int, default=5090,
                    help='seed of the random starting state of the data generation')

parser.add_argument('--batchSize', type=float, default=100,
                    help='batch that the data will be formatted to')

args = parser.parse_args()


testDataLen = int(args.testLen)
seqLen = int(args.seqLen)
seed = args.seed
batchSize = int(args.batchSize)

testSeriesLength = int(testDataLen/batchSize)

stableCoeffs = np.matrix([[ 0,  -0.8],
 [ 0,  -0.6],
 [ 0,  -0.4],
 [ 0,  -0.2],
 [ 0,   0.2],
 [ 0,   0.4],
 [ 0,   0.6],
 [ 0,   0.8],
 [ 0.2, -0.8],
 [ 0.2, -0.6],
 [ 0.2, -0.4],
 [ 0.2, -0.2],
 [ 0.2,  0. ],
 [ 0.2,  0.2],
 [ 0.2,  0.4],
 [ 0.2,  0.6],
 [ 0.2,  0.8],
 [ 0.4, -1. ],
 [ 0.4, -0.8],
 [ 0.4, -0.6],
 [ 0.4, -0.4],
 [ 0.4, -0.2],
 [ 0.4,  0. ],
 [ 0.4,  0.2],
 [ 0.4,  0.4],
 [ 0.6, -0.8],
 [ 0.6, -0.6],
 [ 0.6, -0.4],
 [ 0.6, -0.2],
 [ 0.6,  0. ],
 [ 0.6,  0.2],
 [ 0.8, -0.8],
 [ 0.8, -0.6],
 [ 0.8, -0.4],
 [ 0.8, -0.2],
 [ 0.8,  0. ],
 [ 1,  -1. ],
 [ 1,  -0.8],
 [ 1,  -0.6],
 [ 1,  -0.4],
 [ 1,  -0.2],
 [ 1.2, -0.8],
 [ 1.2, -0.6],
 [ 1.2, -0.4],
 [ 1.4, -1. ],
 [ 1.4, -0.8],
 [ 1.4, -0.6],
 [ 1.6, -1. ],
 [ 1.6, -0.8]])

# stableCoeffs = hdf5s.loadmat('MATfiles/stableCoefficientMatrix.mat')['f1f2Stable']

numCoeffs = stableCoeffs.shape[0]

# trueStateDataTEST Dimensionality: comprised of sets of data based on the number of AR
#                                   coefficients that are to be generated in the 1st
#                                   dimension, then of batch elements in the 2nd
#                                   dimension, real and imaginary portions of the true state
#                                   in the 3rd dimension, and by batches in the 4th dimension
trueStateTEST = np.empty((numCoeffs, batchSize, 4, testSeriesLength), dtype=float)

# measuredStateTEST Dimensionality: comprised of sets of data based on the number of AR
#                                   coefficients used for testing in the 1st dimension,
#                                   batch elements in the 2nd dimension, real and complex
#                                   values in the 3rd dimension, sequence elements in the
#                                   4th dimension, and by batches in the 5th dimension

measuredStateTEST = np.empty((numCoeffs, batchSize, 2, seqLen, testSeriesLength), dtype=float)

LSandKFTestData = []
testDataInfo = []
for k in range(0,numCoeffs):

    # Generating the data with no variance between sequences (which is what the False in the array is for)
    # because we want to generate data with a single set of AR Coefficients
    subsetTestStateData, subsetTestDataInfo = ARDatagenMismatch(params=[testDataLen, 2, 0, seqLen, False], seed=seed + k,
                                                                arCoeffs=stableCoeffs[k])
    trueStateTEST[k,:,:,:], measuredStateTEST[k,:,:,:,:] = convertToBatched(subsetTestStateData[2], subsetTestStateData[1], batchSize)

    # Storing the data that the Least Squares and Kalman Filter will be using
    LSandKFTestData.append(subsetTestStateData)

    subsetInfoHolder = {}
    # Grabbing the first riccatiConvergence because they should all be the same for both the estimate and prediction
    subsetInfoHolder[u'riccatiConvergencePred'] = subsetTestDataInfo['riccatiConvergences'][0,0]
    subsetInfoHolder[u'riccatiConvergenceEst'] = subsetTestDataInfo['riccatiConvergences'][1,0]
    # Grabbing the first set of AR Coefficients from the F matrix because they should all be the same
    subsetInfoHolder[u'ARCoefficients'] = subsetTestDataInfo['allF'][0,:,0]
    # Grabbing the file path of the data file
    subsetInfoHolder[u'dataFilePath'] = subsetTestDataInfo['filename']
    subsetInfoHolder[u'seed'] = subsetTestDataInfo['seed']
    testDataInfo.append(subsetInfoHolder)
# Saving relevant data so it can be recovered and reused
testDataToBeSaved = {}
testDataToBeSaved[u'trueStateTEST'] = trueStateTEST
testDataToBeSaved[u'measuredStateTEST'] = measuredStateTEST
testDataToBeSaved[u'testDataInfo'] = testDataInfo
testDataToBeSaved[u'LSandKFTestData'] = LSandKFTestData
testFile = matSave('data', 'linearTestData', testDataToBeSaved)
