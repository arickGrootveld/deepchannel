import numpy as np

def batchingTestData(trueStates, obsStates, batchSize):
    trueStatesShape = trueStates.shape
    obsStatesShape = obsStates.shape
    # Calculating the number of batches from the data.
    # We cut off the remaining data after fitting the maximum
    # number of batches that evenly divide the dataset
    numBatches = int(np.floor(trueStatesShape[2] / args.batchSize))
    
    
    trueStatesBatched = np.zeros((trueStatesShape[0], batchSize, \
                                  trueStatesShape[1], numBatches))

    obsStatesBatched = np.zeros((obsStatesShape[0], batchSize, \
                                obsStatesShape[1], obsStatesShape[2], \
                                numBatches))

    for i in range(0, numBatches):
        for m in range(0, batchSize):
            trueStatesBatched[:, m, :, i] = trueStates[:, :, (batchSize * i) + m]

            obsStatesBatched[:, m, :, :, i] = obsStates[:, :, :, (batchSize * i) + m]
    
    return(trueStatesBatched, obsStatesBatched)

if __name__ == '__main__':
    import hdf5storage as h5
    import argparse
    import os

    import sys
    sys.path.append('..')
    from utilities import matSave
    parser = argparse.ArgumentParser(description="""Reformatting .mat data 
                                                    generated
                                                    by matlab to work
                                                    with our TCN code""")
    parser.add_argument('--targFile', type=str, default='None', metavar='t',
                        help='The file to reformat and save over')

    parser.add_argument('--testData', action='store_true',
                        help='convert to test data (default: False)')

    parser.add_argument('--batchSize', type=int, default=50, metavar='N',
                        help='The size of the batches generated (default: 50)')


    args = parser.parse_args()

    targetFile = args.targFile

    print('Loading data from: ', targetFile)

    matlabFormattedData = h5.loadmat(targetFile)

    runData = dict()

    # If the data is already reformatted, then we can simply inform the user
    # that this is the case and leave the data alone
    if 'reformattedData' in matlabFormattedData.keys():
        runData = matlabFormattedData
        print('data in: ' + targetFile + ' is already reformatted')
    else:

        # os.remove(targetFile)

        # If we are generating test data, then we need to have a slightly
        # different format for the data
        if(args.testData):
            runData['reformattedData'] = True

            # Rewriting the data to a test data file
            #inter1 = targetFile.rfind('/')
            #inter2 = targetFile.find('ManTargData')
            #inter3 = targetFile.rfind('.')
            #targetFile = targetFile[0:inter1 + 1] + 'ManTargTestData' \
            #    + targetFile[inter2 + 11:inter3] + '.mat'
            
            allTrueStates = matlabFormattedData['data']['systemStates']
            # inter1 = LSAndKFTestData.shape
            # runData['LSandKFTestData'] = np.reshape(LSAndKFTestData, (1, inter1[0], \
                                                    # inter1[1], inter1[2]))

            measInter1 = matlabFormattedData['data']['observedStates']
            inter2 = measInter1.shape
            # runData['measuredStateTEST'] = np.reshape(inter1, (1, inter2[0], inter2[1], \
                                                    #   inter2[2]))
            measStateTest = np.reshape(measInter1, (1, inter2[1], inter2[2], \
                                                    inter2[3]))

            trueInter1 = matlabFormattedData['data']['finalStateValues']
            inter2 = trueInter1.shape
            # runData['trueStateTEST'] = np.reshape(inter1, (1, inter2[0], \
                                                #  inter2[1]))
            trueStateTest = np.reshape(trueInter1, (1, inter2[1], \
                                                inter2[2]))

            # Before batching it, we save it for the LSandKFTestData variable
            runData['LSandKFTestData'] = [[allTrueStates, measInter1, trueInter1]]

            [trueStateTest, measStateTest] = batchingTestData(trueStateTest, \
                                                            measStateTest, \
                                                            args.batchSize)

            runData['trueStateTEST'] = trueStateTest
            runData['measuredStateTEST'] = measStateTest
            
            # TODO: Get these numbers correct with Riccati Convergences
            # TODO: being the genie KF MSE's
            # TODO: and the seed being the seed of the data generation process
            inter1 = dict()
            inter1['riccatiConvergenceEst'] = 0.00
            inter1['riccatiConvergencePred'] = 0.00
            inter1['seed'] = matlabFormattedData['data']['seed'][0,0,0]
            runData['testDataInfo'] = [inter1]
            
            matSave('data', 'ManTargTestData', runData)

        else:
 
            runData['finalStateValues'] = matlabFormattedData['data']['finalStateValues']
            runData['observedStates'] = matlabFormattedData['data']['observedStates']
            runData['systemStates'] = matlabFormattedData['data']['systemStates']

            # Intermediate variable to hold the parameters so they can be 
            # reorganized in usable fashion
            intermed = dict()
            intermed['numSequences'] = matlabFormattedData['data']['parameters'][0,0]['numSequences'][0,0][0,0]
            intermed['sequenceLength'] = matlabFormattedData['data']['parameters'][0,0]['sequenceLength'][0,0][0,0]
            intermed['seed'] = matlabFormattedData['data']['seed'][0,0][0,0]
            # Saving the parameters to data to be saved
            runData['parameters'] = intermed

            runData['riccatiConvergences'] = matlabFormattedData['data']['riccatiConvergences'][0][0]
            runData['reformattedData'] = True

            # Saving the data to its new file
            matSave('data', 'ManTargData', runData)



        

