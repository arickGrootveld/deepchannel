import sys
sys.path.append('..')

import numpy as np
import argparse
import hdf5storage as hdf5s
import os.path as path
import time
from utilities import matSave



start = time.time()

parser = argparse.ArgumentParser(description='Kalman Filter implementation that tests the mean squared '
                                             'error of a KF that has mismatched parameters from '
                                             'the actual AR process')

# File path to load data from
parser.add_argument('--filePath', type=str, default='None',
                    help='path to .mat file to load data from (default: generate data to test against')

# AR Coefficients that the Kalman filter is given
# TODO: Throw an exception if the length of the ARCoeffs list is not 2
parser.add_argument('--ARCoeffs', nargs='+', default=[0.5,0.4],
                     help='AR Coefficients that Kalman Filter will use (default=[0.5,0.4])')
# TODO: As of right now this is not synced up with the data generated, so you need to manually
# TODO: update this in both places if you want to generate fresh data with different AR params

args = parser.parse_args()
ARCoeffs = []
for coeffs in args.ARCoeffs:
    ARCoeffs.append(float(coeffs))
AR_n = len(ARCoeffs)





# If no file was passed to it, will just generate its own data to test against
if args.filePath == 'None':
    from mismatch_data_gen import ARDatagenMismatch

    defaultDataGenValues = {}
    defaultDataGenValues[u'simLength'] = 4000
    defaultDataGenValues[u'AR_n'] = AR_n
    defaultDataGenValues[u'coefVariance'] = 0.2
    defaultDataGenValues[u'batchSize'] = 32
    defaultDataGenValues[u'dataLength'] = 20
    defaultDataGenValues[u'seed'] = 10
    defaultDataGenValues[u'cuda'] = True

    dataGenParameters = [defaultDataGenValues['simLength'],
                     defaultDataGenValues['AR_n'],
                     defaultDataGenValues['coefVariance'],
                     defaultDataGenValues['batchSize'],
                     defaultDataGenValues['dataLength']]


    trueStateData, measuredStateData = ARDatagenMismatch(dataGenParameters, defaultDataGenValues['seed'], cuda=defaultDataGenValues['cuda'])

# If a file path was passed to it, load the data from the file. Expects it to be formatted how
# ARDatagenMismatch formats it
else:
    # Throw error if filepath could not be found
    if(not path.exists(args.filePath)):
        raise Exception('file path: "{}" could not be found'.format(args.filePath))

    matData = hdf5s.loadmat(args.filePath)
    measuredStateData = matData['measuredData']
    trueStateData = matData['predAndCurState']
    print('loaded from file: ', args.filePath)

##### Kalman Filter Implementation #####
# Initializing the Kalman Filter variables
# For a better understanding of the vocabulary used in this code, please consult the mismatch_data_gen
# file, above the ARDatagenMismatch function

# Prediction/estimate formatted into batch elements in the 1st dimension, current and
# last state value in the 2nd dimension, by sequence in the 3rd dimension, and by series 
# in the 4th dimension
x_correction = np.zeros((measuredStateData.shape[0],AR_n,measuredStateData.shape[2],
                         measuredStateData.shape[3]), dtype=complex)
x_prediction = np.zeros((measuredStateData.shape[0],AR_n,measuredStateData.shape[2],
                         measuredStateData.shape[3]), dtype=complex)

# Other Kalman Filter variables formatted into batch elements in the 1st dimension
# a matrix/vector of values in the 2nd and 3rd dimensions, by sequence in the 4th dimension,
# and by series in the 5th dimension
kalmanGain = np.zeros((measuredStateData.shape[0],AR_n, 1, measuredStateData.shape[2],
                       measuredStateData.shape[3]))
minPredMSE = np.zeros((measuredStateData.shape[0],AR_n, AR_n, measuredStateData.shape[2],
                       measuredStateData.shape[3]))
minMSE = np.zeros((measuredStateData.shape[0],AR_n, AR_n, measuredStateData.shape[2],
                   measuredStateData.shape[3]))



# Initializing the correction value to be the expected value of the starting state
x_correction[0,:,0,0] = np.array([0,0])
# Initializing the MSE to be the variance in the starting value of the sequence
minMSE[0,:,:,0,0] = np.array([[0,0], [0,0]])


## Kalman Filter parameters to be used

# F matrix is made up of AR process mean values
F = np.array([ARCoeffs,[1,0]])

# Covariance of the process noise
Q = np.array([[0.1, 0],[0,0]])

# Observation Matrix mapping the observations into the state space
H = np.array([1,0])[np.newaxis]

# Covariance of the measurements
R = np.array([0.1])

# Initializing the total MSE that we will use to follow the MSE through each iteration
totalTruePredMSE = 0
finalTruePredMSE = 0
totalTrueEstimateMSE = 0
finalTrueEstimateMSE = 0

# Loop through the series of data
for i in range(0,measuredStateData.shape[3]):
    # Loop through the batch of data
    for k in range(0,measuredStateData.shape[0]):
        # Loop through a sequence of data
        for q in range(1,measuredStateData.shape[2]):
            # Formatting the measured data properly
            measuredDataComplex = measuredStateData[k,0,q,i] + (measuredStateData[k,1,q,i] *1j)

            # Calculating the prediction of the next state based on the previous estimate
            x_prediction[k,:,q,i] = np.matmul(F, x_correction[k,:,q-1,i])


            # Calculating the predicted MSE from the current MSE, the AR Coefficients,
            # and the covariance matrix
            minPredMSE[k,:,:,q,i] = np.matmul(np.matmul(F,minMSE[k,:,:,q-1,i]), np.transpose(F)) + Q

            # Calculating the new Kalman gain
            intermediate1 = np.matmul(minPredMSE[k,:,:,q,i], np.transpose(H))
            # Intermediate2 should be a single dimensional number, so we can simply just divide by it
            intermediate2 = np.linalg.inv(R + np.matmul(np.matmul(H, minPredMSE[k,:,:,q,i]),
                                                        np.transpose(H)))
            kalmanGain[k,:,:,q,i] = np.matmul(intermediate1, intermediate2)


            # Calculating the State Correction Value
            intermediate1 = np.matmul(H,x_prediction[k,:,q,i])
            intermediate2 = measuredDataComplex - intermediate1
            x_correction[k,:,q,i] = x_prediction[k,:,q,i] + np.matmul(kalmanGain[k,:,:,q,i],
                                                                          intermediate2)

            # Calculating the MSE of our current state estimate
            intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain[k,:,:,q,i], H)
            minMSE[k,:,:,q,i] = np.matmul(intermediate1, minPredMSE[k,:,:,q,i])
        ## Calculating the actual MSE between the kalman filters final prediction, and the actual value ##
        # Converting the true states into their complex equivelants
        currentTrueStateComplex = trueStateData[k,0,i] + (1j* trueStateData[k,2,i])
        nextTrueStateComplex = trueStateData[k,1,i] + (1j*trueStateData[k,3,i])

        finalPrediction = np.matmul(F, x_correction[k,:,q,i])[0]
        finaEstimate = x_correction[k,:,q,i][0]

        # Calculating the instantaneous MSE of our estimate and prediction
        trueEstimateMSE = np.absolute(finaEstimate - currentTrueStateComplex) ** 2
        truePredictionMSE = np.absolute(finalPrediction - nextTrueStateComplex) ** 2

        totalTrueEstimateMSE += trueEstimateMSE
        totalTruePredMSE += truePredictionMSE


    # Averaging the MSE over the batch, and then printing it before reseting it
    totalTrueEstimateMSE = totalTrueEstimateMSE/(trueStateData.shape[0])
    totalTruePredMSE = totalTruePredMSE/(trueStateData.shape[0])
    # print('total MSE of estimate: ', totalTrueEstimateMSE)
    # print('total MSE of prediction: ', totalTruePredMSE)
    finalTrueEstimateMSE += totalTrueEstimateMSE
    finalTruePredMSE += totalTruePredMSE
    totalTrueEstimateMSE = 0
    totalTruePredMSE = 0

finalTrueEstimateMSE =  finalTrueEstimateMSE/(measuredStateData.shape[3])
finalTruePredMSE = finalTruePredMSE/(measuredStateData.shape[3])

print('averaged over everything our estimate MSE is: ', finalTrueEstimateMSE)
print('averaged over everything our prediction MSE is: ', finalTruePredMSE)

end = time.time()

elapsedTime = (end - start)
print('simulation took: ', elapsedTime, ' seconds')

logData = {}

logData[u'predictionMSE'] = finalTruePredMSE
logData[u'estimatedMSE'] = finalTrueEstimateMSE
logData[u'elapsedTime'] = elapsedTime
logData[u'kalmanPredictions'] = x_prediction
logData[u'kalmanEstimates'] =  x_correction
logData[u'kalmanPredMMSE'] = minPredMSE
logData[u'kalmanEstMMSE'] = minMSE
logData[u'kalmanGains'] =  kalmanGain

if not args.filePath == 'None':
    logData[u'dataFileName'] = args.filePath

else:
    logData[u'defaultDataGenValues'] = defaultDataGenValues

matSave('logs', 'KFLog', logData)
