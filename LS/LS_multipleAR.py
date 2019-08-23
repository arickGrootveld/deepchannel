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
parser.add_argument('--filePath', type=str, default='None',
                    help='path to .mat file to load data from')

args = parser.parse_args()


# Throw error if filepath could not be found
if(not path.exists(args.filePath)):
    raise Exception('file path: "{}" could not be found'.format(args.filePath))

# Load relevant data
matData = hdf5s.loadmat(args.filePath)
ARValues = matData['allTrueStateValues']
measuredStateData = matData['measuredData']
trueStateData = matData['predAndCurState']
print('loaded from file: ', args.filePath)

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

intermediate = np.matmul(np.linalg.inv((np.matmul(np.transpose(z), z))), np.transpose(z))

# a - estimate coefficients
a_ls = np.matmul(intermediate, x_est)

# b - prediction coefficients
b_ls = np.matmul(intermediate, x_pred)







# TODO -> different x_est and a_ls/b_ls for testing?

# Calculate MSE of estimation
f = abs((x_est - np.matmul(z, a_ls))) ** 2
MSEE = np.mean(f)

# Calculate MSE of prediction
f = abs((x_pred - np.matmul(z, b_ls))) ** 2
MSEP = np.mean(f)

print("MSEE Avg: ")
print(MSEE)
print("MSPE Avg: ")
print(MSEP)

#     # Grab segments of the AR process and format into a (M x N) matrix
#     z[M-1-j, :] = np.flip(measuredStateDataComplex[j:N+j])
#     #z_pred[M-1-j, :] = np.flip(measuredStateDataComplex[j+i:N+j+i])
#
# # Grab the real state values
# x_est[0:M, 0] = np.flip(ARValuesComplex[N:M+N], 0)
# x_pred[0:M, 0] = np.flip(ARValuesComplex[N:M+N], 0)

















# TODO this code isn't right, training a sequence is not iterative
#
# # Iterate through the sequence
# for i in range(0, measuredStateDataComplex.shape[0]-(M+N-2)-1):
#
#     # Iterate through the number of LS equations
#     for j in range(0, M):
#
#         # Grab segments of the AR process and format into a (M x N) matrix
#         z[M-1-j, :] = np.flip(measuredStateDataComplex[j+i:N+j+i])
#         #z_pred[M-1-j, :] = np.flip(measuredStateDataComplex[j+i:N+j+i])
#
#     # Grab the real state values
#     x_est[0:M, 0] = np.flip(ARValuesComplex[i+N:i+M+N], 0)
#     x_pred[0:M, 0] = np.flip(ARValuesComplex[i+N:i+M+N], 0)
#
#     # Calculate both sets of filter coefficients
#     intermediate = np.matmul(np.linalg.inv((np.matmul(np.transpose(z), z))), np.transpose(z))
#
#     # a - estimate
#     a_ls = np.matmul(intermediate, x_est)
#
#     # b - prediction
#     b_ls = np.matmul(intermediate, x_pred)
#
#     avg = 1/(M+1)
#
# #TODO verify this works -> errors might exist in indexing and finding the MSE values themselves (abs part)
#
#     # MSE Estimation


# TODO THIS IS THE CALCULATION WITH COMPLEX MSE RESULT
#     f = abs((x_est - np.matmul(z, a_ls))) ** 2
#     mean = avg(f)


















##############################################################################################


#     MSEE = np.abs(avg*(np.matmul(np.transpose(f), f)))
#     MSEE_Avg += MSEE
#
#     # MSE Prediction
#     f = (x_pred - np.matmul(z, b_ls))
#     MSPE = np.abs(avg*(np.matmul(np.transpose(f), f)))
#     MSPE_Avg+=MSPE
#
#
#
# MSEE_Avg = MSEE_Avg/(measuredStateDataComplex.shape[0]-(M+N-1))
# MSPE_Avg = MSPE_Avg/(measuredStateDataComplex.shape[0]-(M+N-1))
#
# print("MSEE Avg: ")
# print(MSEE_Avg)
# print("MSPE Avg: ")
# print(MSPE_Avg)
#
#
# # Iterate through the number of LS equations
# for j in range(0, M):
#
#     # Grab segments of the AR process and format into a (M x N) matrix
#     z[M-1-j, :] = np.flip(measuredStateDataComplex[j+994:N+j+994])
#     #z_pred[M-1-j, :] = np.flip(measuredStateDataComplex[j+i:N+j+i])
#
# x_pred[0:M, 0] = np.flip(ARValuesComplex[996:1001], 0)
#
# # MSE Estimation
# f = (x_pred - np.matmul(z, b_ls))
# MSPE = np.abs(avg*(np.matmul(np.transpose(f), f)))
#
# ass = 0




###################################################################################




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
