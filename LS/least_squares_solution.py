# -*- coding: utf-8 -*-


from numpy.linalg import inv
import numpy as np

def least_squares_estimator_predictor(Xhat, Y, state, M, N):
    # inputs - Xhat, Y, state, N, M
    # outputs - a_ls, MSE

    Y1 = Y[:,state]
    Xhat1 = Xhat[0:M, state]
    Ymat = np.zeros((M,N+1))
    
    k = 1
    for j in np.arange(N+1):
        Ymat[:,j] = Y1[k+1: k+M]
        k = k + 1
    
    a_ls =  (inv(np.transpose(Ymat)*Ymat))*(np.transpose(Ymat)*Xhat1)
    MSE = 1/(M+1) * np.transpose((Xhat1 - Ymat*a_ls))*(Xhat1 - Ymat*a_ls)
    
    return a_ls, MSE


N = 10 # Truncation order of an IIR filter
M = 30 # Least Squares order
a_ls_1 = np.zeros((N,M))
num_states = 4
for i in np.arange(num_states):
    a_ls_1, MSE1[i] = least_squares_estimator_predictor(Xhat[i], Y[:,i], i, M, N)
