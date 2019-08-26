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
N = 50

# M - length of observation vector/number of LS equations
M = 700000

# Pre-allocating the matrix that will store the measured data
z = np.zeros((M, N), dtype=complex)

# Pre-allocating the vector that will store the estimations/predictions
x_est = np.zeros((M,1), dtype=complex)
x_pred = np.zeros((M,1), dtype=complex)

######################################################################################
############################## LEAST SQUARES COMPUTATIONS ############################

# Turn observations into complex form from separated real and imaginary components
measuredStateDataComplex = measuredStateData[:,0,:] + (measuredStateData[:,1,:]*1j)
measuredStateDataComplex = np.squeeze(measuredStateDataComplex)

ARValues = ARValues[:, 0, :, :, :]
ARValuesComplex = ARValues[:, 0, :, :] + (ARValues[:, 1, :, :]*1j)
ARValuesComplex = np.squeeze(ARValuesComplex)


# Iterate through the number of LS equations
for j in range(0, M):

    # Grab segments of the AR process and format into a (M x N) matrix
    z[M-1-j, :] = np.flip(measuredStateDataComplex[j:N+j])

    # # DEBUGGING
    #x_est[M-1-j, 0] = measuredStateDataComplex[N-1+j]
    #x_pred[M-1-j, 0] = measuredStateDataComplex[N+j]

    # Grab the real state values
    x_est[(M-1)-j, 0] = ARValuesComplex[N-1+j]
    x_pred[(M-1)-j, 0] = ARValuesComplex[N+j]


# Calculate both sets of filter coefficients
intermediate = np.linalg.pinv(z)

# a - estimate
a_ls = np.matmul(intermediate, x_est)

# b - prediction
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


# Iterate through the number of LS equations
for j in range(0, M):

    # Grab segments of the AR process and format into a (M x N) matrix
    z[M-1-j, :] = np.flip(measuredStateDataComplex[j:N+j])

    # # DEBUGGING
    #x_est[M-1-j, 0] = measuredStateDataComplex[N-1+j]
    #x_pred[M-1-j, 0] = measuredStateDataComplex[N+j]

    # Grab the real state values
    x_est[(M-1)-j, 0] = ARValuesComplex[N-1+j]
    x_pred[(M-1)-j, 0] = ARValuesComplex[N+j]


# Calculate MSE of estimation
f = abs(np.square(x_est - np.matmul(z, a_ls)))
MSEE = np.mean(f)

# Calculate MSE of prediction
f = abs(np.square(x_pred - np.matmul(z, b_ls)))
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

matSave("logs", "lsSingular", MSEVals)
#       directory (str) - the directory for the data to be saved in
#       basename (str) - the name of the file you want the data saved to before the appended number
#       data (dict) - a dict of data to be saved to the .mat file

