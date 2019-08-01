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

# If no file was passed to it, will just generate its own data to test against
if args.filePath == 'None':
    from mismatch_data_gen import ARDatagenMismatch

    defaultDataGenValues = {}
    defaultDataGenValues[u'simLength'] = 10
    defaultDataGenValues[u'AR_n'] = 2
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

# Prediction/estimate formatted into a real and imaginary in the 1st dimension, vectorized in the 2nd
# dimension, by sequence in the 3rd dimension, by batch in the 4th dimension, and by series in the 5th
# dimension
x_correction = np.zeros((2,2,measuredStateData.shape[2],measuredStateData.shape[0],
                         measuredStateData.shape[3]))
x_prediction = np.zeros((2,2,measuredStateData.shape[2],measuredStateData.shape[0],
                         measuredStateData.shape[3]))

# Other Kalman Filter variables formatted in a matrix/vector of values in the 1st  and 2nd dimensions,
# by sequence in the 3rd dimension, by batch in the 4th dimension, and by series in the 5th dimension
kalmanGain = np.zeros((2,1, measuredStateData.shape[2], measuredStateData.shape[0],
                       measuredStateData.shape[3]))
minPredMSE = np.zeros((2,2, measuredStateData.shape[2], measuredStateData.shape[0],
                       measuredStateData.shape[3]))
minMSE = np.zeros((2,2, measuredStateData.shape[2], measuredStateData.shape[0],
                   measuredStateData.shape[3]))



# Initializing the correction value to be the expected value of the starting state
x_correction[:,:,0,0,0] = np.array([[0,0],[0,0]])
# Initializing the MSE to be the variance in the starting value of the sequence
minMSE[:,:,0,0,0] = np.array([[0,0], [0,0]])


## Kalman Filter parameters to be used

# F matrix is made up of AR process mean values
F = np.array([ARCoeffs,[1,0]])

# system noise mean value
sysNoiseMean = np.array([[0,0], [0,0]])

# Covariance of the process noise
Q = np.array([[0.1, 0],[0,0]])

# Observation Matrix mapping the observations into the state space
H = np.array([1,0])

# Covariance of the measurements
R = 0.1


# Loop through the series of data
for i in range(0,measuredStateData.shape[3]):
    print('iteration')
    # Loop through the batch of data
    for k in range(0,measuredStateData.shape[0]):
        print('iteration through a batch')
        # Loop through a sequence of data
        for q in range(1,measuredStateData.shape[2]):
            print('iteration through a sequence')
            # Calculating the prediction of the next state based on the previous estimate
            x_prediction[:,:,q,k,i] = np.matmul(F, x_correction[:,:,q-1,k,i]) + sysNoiseMean
            # Calculating the predicted MSE from the current MSE, the AR Coefficients,
            # and the covariance matrix
            minPredMSE[:,:,q,k,i] = np.matmul(np.matmul(F,minMSE[:,:,q-1,k,i]), np.transpose(F)) + Q
            intermediate1 = np.matmul(minPredMSE[:,:,q,k,i], H)


print('test complete')
print(args)