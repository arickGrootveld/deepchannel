from numpy.linalg import inv
import numpy as np
import time
import argparse
import os.path as path
import hdf5storage as hdf5s
from utilities import matSave

#####################################################################################
######################### Least Squares Channel Estimation ##########################

# Start timing
start = time.time()

# Argument parser
parser = argparse.ArgumentParser(description='Least Squares implementation that predicts and estimates '
                                             'channel states and calculates the MSE of these values')

# File path to load data from
parser.add_argument('--filePathTrain', type=str, default='None',
                    help='path to .mat file to load training data from')

parser.add_argument('--filePathTest', type=str, default='None',
                    help='path to .mat file to load test data from')

args = parser.parse_args()


# Throw error if filepath could not be found
if(not path.exists(args.filePathTrain)):
    raise Exception('file path: "{}" could not be found'.format(args.filePathTrain))

# Load relevant training data
matData = hdf5s.loadmat(args.filePathTrain)
ARValues = matData['allTrueStateValues']
measuredStateData = matData['measuredData']
trueStateData = matData['predAndCurState']
print('loaded from file: ', args.filePathTrain)

# The least squares algorithm uses the following matrix multiplication to achieve the predictions and estimations

#  predications/observations             measurements       filter taps
#           (M x 1)             =          (M x N)      *     (N x 1)

# N - length of the history/filter taps
N = measuredStateData.shape[2]

# M - length of observation vector/number of LS equations
M = measuredStateData.shape[0]*measuredStateData.shape[3]

# Pre-allocating the matrix that will store the measured data
z = np.zeros((M, N), dtype=complex)
#z_pred = np.zeros((M, N), dtype=complex)

# Pre-allocating the vector that will store the estimations/predictions
x_est = np.zeros((M,1), dtype=complex)
x_pred = np.zeros((M,1), dtype=complex)

# Turn observations into complex form from separated real and imaginary components
measuredStateDataComplex = measuredStateData[:,0,:, :] + (measuredStateData[:,1,:, :]*1j)
measuredStateDataComplex = np.squeeze(measuredStateDataComplex)

# Turn real state values into complex form from separated real and imaginary components
ARValues = ARValues[:, 0, :, :, :]
ARValuesComplex = ARValues[:, 0, :, :] + (ARValues[:, 1, :, :]*1j)
ARValuesComplex = np.squeeze(ARValuesComplex)

######################################################################################
############################## LEAST SQUARES COMPUTATIONS ############################

# Iterate through the series
for i in range(0, ARValuesComplex.shape[2]):

    # Iterate through the batch
    for j in range (0, ARValuesComplex.shape[0]):

        # Constructing the observation matrix
        z[i*ARValuesComplex.shape[0]+j, :] = np.flipud(measuredStateDataComplex[j, 0:N, i])

        # Construct the estimation and prediction real states for MSE calculation and training
        x_est[i*ARValuesComplex.shape[0]+j, :] = ARValuesComplex[j, ARValuesComplex.shape[1]-2, i]
        x_pred[i * ARValuesComplex.shape[0] + j, :] = ARValuesComplex[j, ARValuesComplex.shape[1]-1, i]


# Singular matrix problem fix
z = z[:, 0:19]

intermediate = np.linalg.pinv(z)

# a - estimate coefficients
a_ls = np.matmul(intermediate, x_est)

# b - prediction coefficients
b_ls = np.matmul(intermediate, x_pred)



##############################################################################################
################################ LEAST SQUARES TESTING SCRIPT ################################


# Throw error if filepath could not be found
if(not path.exists(args.filePathTest)):
    raise Exception('file path: "{}" could not be found'.format(args.filePathTest))

# Load relevant training data
matData = hdf5s.loadmat(args.filePathTest)
ARValues = matData['allTrueStateValues']
measuredStateData = matData['measuredData']
trueStateData = matData['predAndCurState']
print('loaded from file: ', args.filePathTest)



# Turn observations into complex form from separated real and imaginary components
measuredStateDataComplex = measuredStateData[:,0,:] + (measuredStateData[:,1,:]*1j)
measuredStateDataComplex = np.squeeze(measuredStateDataComplex)

ARValues = ARValues[:, 0, :, :, :]
ARValuesComplex = ARValues[:, 0, :, :] + (ARValues[:, 1, :, :]*1j)
ARValuesComplex = np.squeeze(ARValuesComplex)

# Pre-allocating the matrix that will store the measured data
z = np.zeros((M, N), dtype=complex)

# Iterate through the series
for i in range(0, ARValuesComplex.shape[2]):

    # Iterate through the batch
    for j in range (0, ARValuesComplex.shape[0]):

        # Constructing the observation matrix
        z[i*ARValuesComplex.shape[0]+j, :] = np.flipud(measuredStateDataComplex[j, 0:N, i])

        # Construct the estimation and prediction real states for MSE calculation and training
        x_est[i*ARValuesComplex.shape[0]+j, :] = ARValuesComplex[j, ARValuesComplex.shape[1]-2, i]
        x_pred[i * ARValuesComplex.shape[0] + j, :] = ARValuesComplex[j, ARValuesComplex.shape[1]-1, i]


# Singular matrix problem fix
z = z[:, 0:19]

# Calculate MSE of estimation
f = abs((x_est - np.matmul(z, a_ls))) ** 2
MSEE = np.mean(f)

# Calculate MSE of prediction
f = abs((x_pred - np.matmul(z, b_ls))) ** 2
MSEP = np.mean(f)

########################################################
# Log and Print Results

print("MSEE Avg: ")
print(MSEE)
print("MSPE Avg: ")
print(MSEP)

MSEVals = {}
MSEVals[u'M'] = M
MSEVals[u'N'] = N
MSEVals[u'MSE_est'] = MSEE
MSEVals[u'MSE_pred'] = MSEP

matSave("logs", "lsMultiple", MSEVals)


