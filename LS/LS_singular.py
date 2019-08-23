from numpy.linalg import inv
import numpy as np
import time
import argparse
import os.path as path
import hdf5storage as hdf5s
import torch
import cmath

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
N = 10

# M - length of observation vector/number of LS equations
M = 500000

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
intermediate = np.matmul(np.linalg.pinv((np.matmul(np.transpose(z), z))), np.transpose(z))

# a - estimate
a_ls = np.matmul(intermediate, x_est)

# b - prediction
b_ls = np.matmul(intermediate, x_pred)

# a_ls = np.transpose(np.matrix([.546845223756256 + 0.108182806879480j, 0.133227098016353 + 0.008626744487407j, -0.080117991483541 - 0.123662858151174j, -0.204860984672930 + 0.203284158366416j,
#                   0.089800564601550 + 0.077211768209860j, 0.160578773433174 + 0.033510337476693j, 0.061665434155100 + 0.106041355547560j,
#                   0.259676378708334 + 0.127092995228038j, -0.001848188584772 - 0.154626193733209j, 0.228794245949007 + 0.083261545696778j], dtype=complex))
#
# b_ls = np.transpose(np.matrix([0.1053 + 0.1479j, -0.3444 - 0.1524j, -0.0662 + 0.4348j, 0.3479 + 0.4943,
#                   0.4902 + 0.0119j, 0.0771 - 0.2991j, 0.0346 + 0.0441j,
#                   0.3030 - 0.3349j, -0.1375 - 0.4279j, -0.3129 + 0.0726j], dtype=complex))

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
f = np.square(abs((x_est - np.matmul(z, a_ls))))
MSEE = np.mean(f)

# Calculate MSE of prediction
f = np.square(abs((x_pred - np.matmul(z, b_ls))))
MSEP = np.mean(f)

print("MSEE Avg: ")
print(MSEE)
print("MSPE Avg: ")
print(MSEP)



# def least_squares_estimator_predictor(Xhat, Y, state, M, N):
#     # inputs - Xhat, Y, state, N, M
#     # outputs - a_ls, MSE
#
#     Y1 = Y[:,state]
#     Xhat1 = Xhat[0:M, state]
#     Ymat = np.zeros((M,N+1))
#
#     k = 1
#     for j in np.arange(N+1):
#         Ymat[:,j] = Y1[k+1: k+M]
#         k = k + 1
#
#     a_ls =  (inv(np.transpose(Ymat)*Ymat))*(np.transpose(Ymat)*Xhat1)
#     MSE = 1/(M+1) * np.transpose((Xhat1 - Ymat*a_ls))*(Xhat1 - Ymat*a_ls)
#
#     return a_ls, MSE
#
#
#
#
#
# N = 10 # Truncation order of an IIR filter
# M = 30 # Least Squares order
# num_states = 4
# state = 1
# Xhat = np.squeeze(predAndCurState[1,state,:])
# Y1 = np.squeeze(trainDataMeas[0:size(Xhat,1), state])
#
# a_ls_1 = np.zeros((N,num_states))
# for i in np.arange(num_states):
#     a_ls_1[i], MSE1[i] = least_squares_estimator_predictor(Xhat[i], Y[:,i], i, M, N)
