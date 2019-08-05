import numpy as np
import argparse
import hdf5storage as hdf5s
import os.path as path

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

ARCoeffs = args.ARCoeffs
AR_n = len(ARCoeffs)

# If no file was passed to it, will just generate its own data to test against
if args.filePath == 'None':
    from mismatch_data_gen import ARDatagenMismatch

    defaultDataGenValues = {}
    defaultDataGenValues[u'simLength'] = 2
    defaultDataGenValues[u'AR_n'] = AR_n
    defaultDataGenValues[u'coefVariance'] = 0.1
    defaultDataGenValues[u'batchSize'] = 32
    defaultDataGenValues[u'dataLength'] = 20

    dataGenValues = [defaultDataGenValues['simLength'],
                     defaultDataGenValues['AR_n'],
                     defaultDataGenValues['coefVariance'],
                     defaultDataGenValues['batchSize'],
                     defaultDataGenValues['dataLength']]

    trueStateData, measuredStateData = ARDatagenMismatch(dataGenValues)

# If a file path was passed to it, load the data from the file. Expects it to be formatted how
# ARDatagenMismatch formats it
else:
    # Throw error if filepath could not be found
    if(not path.exists(args.filePath)):
        raise Exception('file path: "{}" could not be found'.format(args.filePath))

    matData = hdf5s.loadmat(args.filePath)
    measuredStateData = matData['measuredData']
    trueStateData = matData['predAndCurState']
    print(matData['measuredData'])

##### Kalman Filter Implementation #####
# Initializing the Kalman Filter variables
# For a better understanding of the vocabulary used in this code, please consult the mismatch_data_gen
# file, above the ARDatagenMismatch function

# Prediction/estimate formatted into batch elements in the 1st dimension, current and
# last state value in the 2nd dimension, real and complex in the 3rd dimension,
# by sequence in the 4th dimension, and by series in the 5th dimension
x_correction = np.zeros((measuredStateData.shape[0],AR_n,AR_n,measuredStateData.shape[2],
                         measuredStateData.shape[3]))
x_prediction = np.zeros((measuredStateData.shape[0],AR_n,AR_n,measuredStateData.shape[2],
                         measuredStateData.shape[3]))

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
x_correction[0,:,:,0,0] = np.array([[0,0],[0,0]])
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


# Loop through the series of data
for i in range(0,measuredStateData.shape[3]):
    # Loop through the batch of data
    for k in range(0,measuredStateData.shape[0]):
        # Loop through a sequence of data
        for q in range(1,measuredStateData.shape[2]):

            # Calculating the prediction of the next state based on the previous estimate
            x_prediction[k,:,:,q,i] = np.matmul(F, x_correction[k,:,:,q-1,i])


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
            intermediate1 = np.matmul(H,x_prediction[k,:,:,q,i])
            intermediate2 = measuredStateData[k,:,q,i] - intermediate1
            x_correction[k,:,:,q,i] = x_prediction[k,:,:,q,i] + np.matmul(kalmanGain[k,:,:,q,i],
                                                                          intermediate2)

            # Calculating the MSE of our current state estimate
            intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain[k,:,:,q,i], H)
            minMSE[k,:,:,q,i] = np.matmul(intermediate1, minPredMSE[k,:,:,q,i])
    print('sequence done')

# x_correction

## TODO: Verify that this works with live data
## TODO: Do the actual computation of mean squared errors and save that to a .mat file, as well as the average MSE from all sequences
## TODO: Compare this with Neural Network and LS estimator to determine how effective the NN is

print('test complete')
print(args)