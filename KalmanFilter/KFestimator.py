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
parser.add_argument('--ARCoeffs', nargs='+', default=[0.3645,-0.3405],
                     help='AR Coefficients that Kalman Filter will use (default=[0.3645,-0.3405])')
# TODO: As of right now this is not synced up with the data generated, so you need to manually
# TODO: update this in both places if you want to generate fresh data with different AR params
args = parser.parse_args()
ARCoeffs = []

# Because the code is built for an AR 2 process at this time it will throw an error if you specify too many
# AR coefficients
if len(args.ARCoeffs) != 2:
    raise Exception('At this time you can only pass 2 AR coefficients because this code is only'
                    'compatible with an AR 2 process')

for coeffs in args.ARCoeffs:
    ARCoeffs.append(float(coeffs))
AR_n = len(ARCoeffs)


# If no file was passed to it, will just generate its own data to test against
if args.filePath == 'None':
    from mismatch_data_gen import ARDatagenMismatch

    defaultDataGenValues = {}
    defaultDataGenValues[u'simLength'] = 400
    defaultDataGenValues[u'AR_n'] = AR_n
    defaultDataGenValues[u'coefVariance'] = 0
    defaultDataGenValues[u'dataLength'] = 20
    defaultDataGenValues[u'seed'] = 10
    defaultDataGenValues[u'cuda'] = False

    dataGenParameters = [defaultDataGenValues['simLength'],
                     defaultDataGenValues['AR_n'],
                     defaultDataGenValues['coefVariance'],
                     defaultDataGenValues['dataLength']]


    # trueStateData, measuredStateData = ARDatagenMismatch(dataGenParameters, defaultDataGenValues['seed'], cuda=defaultDataGenValues['cuda'])
    stateData, stateInfo = ARDatagenMismatch(dataGenParameters, defaultDataGenValues['seed'], cuda=defaultDataGenValues['cuda'])
    trueStateData = stateData[2]
    measuredStateData = stateData[1]
    all_F = stateInfo['trueFs']

# If a file path was passed to it, load the data from the file. Expects it to be formatted how
# ARDatagenMismatch formats it - also loads all F matrices for best possible MSE value computation
else:
    # Throw error if filepath could not be found
    if(not path.exists(args.filePath)):
        raise Exception('file path: "{}" could not be found'.format(args.filePath))

    matData = hdf5s.loadmat(args.filePath)
    measuredStateData = matData['observedStates']
    trueStateData = matData['finalStateValues']
    all_F = matData['trueFs']
    print('loaded from file: ', args.filePath)

# Variables for deciding length of variables
sequenceLength = measuredStateData.shape[1]
seriesLength = measuredStateData.shape[2]

##### Kalman Filter Implementation #####
# Initializing the Kalman Filter variables
# For a better understanding of the vocabulary used in this code, please consult the mismatch_data_gen
# file, above the ARDatagenMismatch function

# Prediction/estimate formatted into current and last state value in the 1st dimension,
# by sequence in the 2nd dimension, and by series in the 3rd dimension
x_correction = np.zeros((AR_n,sequenceLength,
                         seriesLength), dtype=complex)
x_prediction = np.zeros((AR_n,sequenceLength,
                         seriesLength), dtype=complex)

# Other Kalman Filter variables formatted into a matrix/vector of values in the 1st and 2nd dimensions,
# by sequence in the 3rd dimension, and by series in the 4th dimension
kalmanGain = np.zeros((AR_n, 1, sequenceLength,
                       seriesLength))
minPredMSE = np.zeros((AR_n, AR_n, sequenceLength,
                       seriesLength))
minMSE = np.zeros((AR_n, AR_n, sequenceLength,
                   seriesLength))



# Initializing the correction value to be the expected value of the starting state
x_correction[:,0,0] = np.array([0,0])
# Initializing the MSE to be the variance in the starting value of the sequence
minMSE[:,:,0,0] = np.array([[1,0], [0,1]])


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

############################################################################################
########################### PARALLEL KALMAN FILTER PARAMETERS ##############################

# Dimensions of all parameters are the same as before. Only difference is that now, we know
# the F matrices which are loaded from a saved matlab file.

x_correction_parallel = np.zeros((AR_n,sequenceLength,
                         seriesLength), dtype=complex)
x_prediction_parallel = np.zeros((AR_n,sequenceLength,
                         seriesLength), dtype=complex)

kalmanGain_parallel = np.zeros((AR_n, 1, sequenceLength,
                       seriesLength))
minPredMSE_parallel = np.zeros((AR_n, AR_n, sequenceLength,
                       seriesLength))
minMSE_parallel = np.zeros((AR_n, AR_n, sequenceLength,
                   seriesLength))

# Initializing the correction value to be the expected value of the starting state
x_correction_parallel[:,0,0] = np.array([0,0])
# Initializing the MSE to be the variance in the starting value of the sequence
minMSE_parallel[:,:,0,0] = np.array([[1,0], [0,1]])

# F matrix is made up of AR process mean values
F_parallel = np.array([ARCoeffs,[1,0]])

# Covariance of the process noise
Q_parallel = np.array([[0.1, 0],[0,0]])

# Observation Matrix mapping the observations into the state space
H_parallel = np.array([1,0])[np.newaxis]

# Covariance of the measurements
R_parallel = np.array([0.1])

# Initializing the total MSE that we will use to follow the MSE through each iteration
totalTruePredMSE_parallel = 0
finalTruePredMSE_parallel = 0
totalTrueEstimateMSE_parallel = 0
finalTrueEstimateMSE_parallel = 0

############################################################################################
F_IX = 0
# Loop through the series of data
for i in range(0,seriesLength):

    F_parallel = all_F[:, :, F_IX]
    F_IX += 1

    # Loop through a sequence of data
    for q in range(1,sequenceLength):

        #############################################################################
        ############################# KALMAN FILTER 1  ##############################
        # This is the original Kalman Filter - does not know the actual AR coeffecients of the
        # AR processes that are input, only knows the theoretical mean values.

        # Formatting the measured data properly
        measuredDataComplex = measuredStateData[0,q,i] + (measuredStateData[1,q,i] *1j)

        # Calculating the prediction of the next state based on the previous estimate
        x_prediction[:,q,i] = np.matmul(F, x_correction[:,q-1,i])

        # Calculating the predicted MSE from the current MSE, the AR Coefficients,
        # and the covariance matrix
        minPredMSE[:,:,q,i] = np.matmul(np.matmul(F,minMSE[:,:,q-1,i]), np.transpose(F)) + Q

        # Calculating the new Kalman gain
        intermediate1 = np.matmul(minPredMSE[:,:,q,i], np.transpose(H))
        # Intermediate2 should be a single dimensional number, so we can simply just divide by it
        intermediate2 = np.linalg.inv(R + np.matmul(np.matmul(H, minPredMSE[:,:,q,i]),
                                                    np.transpose(H)))
        kalmanGain[:,:,q,i] = np.matmul(intermediate1, intermediate2)

        # Calculating the State Correction Value
        intermediate1 = np.matmul(H,x_prediction[:,q,i])
        intermediate2 = measuredDataComplex - intermediate1
        x_correction[:,q,i] = x_prediction[:,q,i] + np.matmul(kalmanGain[:,:,q,i],
                                                                      intermediate2)
        # Calculating the MSE of our current state estimate
        intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain[:,:,q,i], H)
        minMSE[:,:,q,i] = np.matmul(intermediate1, minPredMSE[:,:,q,i])

        #############################################################################
        ############################# KALMAN FILTER 2  ##############################
        # This is the parallel Kalman Filter - knows the actual values of the AR coeffecients
        # that are input (this is used to provide a lower bound for performance which we
        # can use to check the network against). The MSE of this Kalman Filter should
        # be nearly the best theoretical that we can achieve.

        # Calculating the prediction of the next state based on the previous estimate
        x_prediction_parallel[:, q, i] = np.matmul(F_parallel, x_correction_parallel[:, q - 1, i])

        # Calculating the predicted MSE from the current MSE, the AR Coefficients,
        # and the covariance matrix
        minPredMSE_parallel[:, :, q, i] = np.matmul(np.matmul(F_parallel, minMSE_parallel[:, :, q - 1, i]), np.transpose(F_parallel)) + Q_parallel

        # Calculating the new Kalman gain
        intermediate1 = np.matmul(minPredMSE_parallel[:, :, q, i], np.transpose(H_parallel))
        # Intermediate2 should be a single dimensional number, so we can simply just divide by it
        intermediate2 = np.linalg.inv(R_parallel + np.matmul(np.matmul(H_parallel, minPredMSE_parallel[:, :, q, i]),
                                                    np.transpose(H_parallel)))
        kalmanGain_parallel[:, :, q, i] = np.matmul(intermediate1, intermediate2)

        # Calculating the State Correction Value
        intermediate1 = np.matmul(H_parallel, x_prediction_parallel[:, q, i])
        intermediate2 = measuredDataComplex - intermediate1
        x_correction_parallel[:, q, i] = x_prediction_parallel[:, q, i] + np.matmul(kalmanGain_parallel[:, :, q, i], intermediate2)
        # Calculating the MSE of our current state estimate
        intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain_parallel[:, :, q, i], H_parallel)
        minMSE_parallel[:, :, q, i] = np.matmul(intermediate1, minPredMSE_parallel[:, :, q, i])

    #############################################################################
    ############################# KALMAN FILTER 1  ##############################

    ## Calculating the actual MSE between the kalman filters final prediction, and the actual value ##
    # Converting the true states into their complex equivalents
    currentTrueStateComplex = trueStateData[0,i] + (1j* trueStateData[2,i])
    nextTrueStateComplex = trueStateData[1,i] + (1j*trueStateData[3,i])

    finalPrediction = np.matmul(F, x_correction[:,q,i])[0]
    finaEstimate = x_correction[:,q,i][0]

    # Calculating the instantaneous MSE of our estimate and prediction
    trueEstimateMSE = np.absolute(finaEstimate - currentTrueStateComplex) ** 2
    truePredictionMSE = np.absolute(finalPrediction - nextTrueStateComplex) ** 2

    totalTrueEstimateMSE += trueEstimateMSE
    totalTruePredMSE += truePredictionMSE

    #############################################################################
    ############################# KALMAN FILTER 2  ##############################

    finalPrediction_parallel = np.matmul(F_parallel, x_correction_parallel[:, q, i])[0]
    finaEstimate_parallel = x_correction_parallel[:, q, i][0]

    # Calculating the instantaneous MSE of our estimate and prediction
    trueEstimateMSE_parallel = np.absolute(finaEstimate_parallel - currentTrueStateComplex) ** 2
    truePredictionMSE_parallel = np.absolute(finalPrediction_parallel - nextTrueStateComplex) ** 2

    totalTrueEstimateMSE_parallel += trueEstimateMSE_parallel
    totalTruePredMSE_parallel += truePredictionMSE_parallel


#############################################################################
############################# KALMAN FILTER 1  ##############################
totalTrueEstimateMSE =  totalTrueEstimateMSE/(seriesLength)
totalTruePredMSE = totalTruePredMSE/(seriesLength)

#############################################################################
############################# KALMAN FILTER 2  ##############################
totalTrueEstimateMSE_parallel =  totalTrueEstimateMSE_parallel/(seriesLength)
totalTruePredMSE_parallel = totalTruePredMSE_parallel/(seriesLength)


print('averaged over everything our estimate MSE is: ', totalTrueEstimateMSE)
print('averaged over everything our prediction MSE is: ', totalTruePredMSE)

print('averaged over everything our BEST POSSIBLE estimate MSE is: ', totalTrueEstimateMSE_parallel)
print('averaged over everything our BEST POSSIBLE prediction MSE is: ', totalTruePredMSE_parallel)

####################################################################
# File logging - time and relevant parameters saved to matlab file #

end = time.time()

elapsedTime = (end - start)
print('simulation took: ', elapsedTime, ' seconds')

logData = {}

logData[u'predictionMSE'] = totalTruePredMSE
logData[u'estimatedMSE'] = totalTrueEstimateMSE

logData[u'BPpredictionMSE'] = totalTruePredMSE_parallel
logData[u'BPestimatedMSE'] = totalTrueEstimateMSE_parallel

logData[u'elapsedTime'] = elapsedTime
logData[u'kalmanPredictions'] = x_prediction
logData[u'kalmanEstimates'] =  x_correction
logData[u'kalmanPredMMSE'] = minPredMSE
logData[u'kalmanEstMMSE'] = minMSE
logData[u'kalmanGains'] =  kalmanGain
logData[u'runType'] = 'KF'

if not args.filePath == 'None':
    logData[u'dataFileName'] = args.filePath
else:
    logData[u'dataFileName'] = stateInfo['filename']

matSave('logs', 'KFLog', logData)
