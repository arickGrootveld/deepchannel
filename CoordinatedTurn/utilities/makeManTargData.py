import numpy as np
import hdf5storage as h5
import argparse

import sys
sys.path.append('..')
from utilities import matSave

parser = argparse.ArgumentParser(description='Script that will stitch together'
                                'the data that matlab generated for the'
                                'manTarg scenario')
parser.add_argument('--seqLen', type=int, default=10, 
                    help='sequence Length of the data to be fed to TCN and LS'
                    '(default: 10)')

parser.add_argument('--testData', action='store_true', 
                    help='Whether this is test data or not (default: False)')

args = parser.parse_args()

testMode = args.testData
# Hardcoded for now
dataInfo = h5.loadmat('data/matData.mat')

seqLen = args.seqLen

# Extracting the data from the various files
trueStateMatData = h5.loadmat(dataInfo['saveData']['trueStateFiles'][0,0][0])

obsStateMatData = h5.loadmat(dataInfo['saveData']['obsStateFiles'][0,0][0])

# Formatting the data to be as the TCN expects it

trueStateData = trueStateMatData['sTrueStates']
obsStateData = obsStateMatData['Y']

# Preallocating the matrices for obsStates, finalStateValues, and trueStates
dataShape = obsStateData.shape
numSamples = dataShape[1] - seqLen

observedStates = np.zeros([dataShape[0], seqLen, numSamples])
systemStates = np.zeros([dataShape[0], seqLen + 1, numSamples])
finalStateValues = np.zeros([2*dataShape[0], numSamples])

# Formatting final state values
for i in range(0, numSamples):
    finalStateValues[0,i] = trueStateData[0, i+seqLen-1]
    finalStateValues[1, i] = trueStateData[0, i + seqLen]
    finalStateValues[2,i] = trueStateData[1, i+seqLen-1]
    finalStateValues[3, i] = trueStateData[1, i + seqLen]

    observedStates[:,:, i] = obsStateData[:, i:i+seqLen]

    systemStates[:, :, i] = trueStateData[:, i:i+seqLen+1]

saveData = dict()
if not testMode:
    print('under construction')

    # Once data formatted correctly, we then save it
    saveData['finalStateValues'] = finalStateValues
    saveData['observedStates'] = observedStates
    saveData['systemStates'] = systemStates

    intermed = dict()
    
    intermed['seed'] = dataInfo['saveData']['seed'][0,0][0,0]
    
    intermed['numSequences'] = numSamples
    intermed['sequenceLength'] = seqLen

    saveData['parameters'] = intermed
    saveData['riccatiConvergences'] = dataInfo['saveData']['riccatiConvergences'][0,0]
    
    matSave('data', 'ManTargData', saveData)


else:
    testBatchSize = 10
    numBatches = int(np.floor(observedStates.shape[2] / testBatchSize))

    measuredStateTest = np.zeros([1, testBatchSize, 
                                  dataShape[0], seqLen, numBatches])

    trueStateTEST = np.zeros([1, testBatchSize, 2*dataShape[0], numBatches])

    testDataInfo = []
    testDataInfoDict = dict()
    testDataInfoDict['seed'] = dataInfo['saveData']['seed'][0,0][0,0]
    testDataInfoDict['riccatiConvergencePred'] = 0
    testDataInfoDict['riccatiConvergenceEst'] = 0

    # Batching the data
    for i in range(0, numBatches):
        trueStateTEST[0, :, :, i] = np.transpose(finalStateValues[:, 
                                    i*testBatchSize:i*testBatchSize 
                                    + testBatchSize])
        measuredStateTest[0, :, :, :, i] = np.reshape(observedStates[:,:, 
                                           i*testBatchSize:i*testBatchSize 
                                           + testBatchSize], 
                                           measuredStateTest[0, :, :, :, i]
                                           .shape)



    saveData['LSandKFTestData'] = [[systemStates, observedStates, finalStateValues]]
    
    saveData['trueStateTEST'] = trueStateTEST
    saveData['measuredStateTEST'] = measuredStateTest

    testDataInfo[0] = testDataInfoDict
    saveData['testDataInfo'] = testDataInfo

    matSave('data', 'ManTargTestData', saveData)


print('worked?')