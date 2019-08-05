# Generate an AR process with mismatched AR coefficients - for training and evaluation of model

import numpy as np
from numpy import linalg as LA
from utilities import matSave
# import hdf5storage as hdf5s
# import os
# import os.path as path


# ARCoeffecientGeneration: Function that returns a matrix that works as an AR processes F matrix
#   Inputs:  (arCoeffMeans, arCoefficientNoiseVar)
#       arCoeffMeans (array) - mean values of the AR coefficients to be generated
#           arCoeffMeans[0] (float) - first AR coefficient
#           arCoeffMeans[1] (float) - second AR coefficient
#       arCoefficientNoiseVar (float) - variance of the AR coefficients to be generated
#   Outputs: (arCoeffMatrix)
#       arCoeffMatrix (tensor [2 x 2]) - the F matrix of an AR process (Bar-Shalom's notation)
#                                                                   
def ARCoeffecientGeneration(arCoeffMeans,arCoeffecientNoiseVar, seed=-1):
    if(seed > 0):
        np.random.seed(seed)

    arCoeffsMatrix = np.identity(2)
    arCoeffsMatrix[1] = arCoeffsMatrix[0]
    goodCoeffs = False
    # Pre-Allocating the arCoeffNoise array
    arCoeffNoise = [0,0]
    while(not goodCoeffs):
        # Generate new AR Coefficients until you get a pair who's eigenvalues are 
        # less than 1.
        # We do this because the system would explode to infinity if the eigenvalues 
        # were greater than 1.
        arCoeffNoise[0] = np.random.randn(1) * arCoeffecientNoiseVar
        arCoeffNoise[1] = np.random.randn(1) * arCoeffecientNoiseVar
        arCoeffsMatrix[0,0] = arCoeffMeans[0] + arCoeffNoise[0]
        arCoeffsMatrix[0,1] = arCoeffMeans[1] + arCoeffNoise[1]

        # Compute EigenValues of F
        eVals, eVecs = LA.eig(arCoeffsMatrix)
        # Determine if any of them have a greater magnitude than 1
        if (not np.any(np.absolute(eVals)>1)):
            goodCoeffs=True
    return arCoeffsMatrix



# ARDatagenMismatch: Function that returns a set of data with a specified length, that
#                    comes from an AR process with AR coefficients that have some small
#                    amount of noise added to them
#   Inputs: (params, seed)
#       params (list) - a set of parameters to pass to the model
#           params[0] (int) - simLength: amount of batches of data to be generated by the
#                                        simulation
#           params[1] (int) - AR_n: The AR process order (The number of AR coefficients)
#           params[2] (float) - AR_coefficient_noise_var: variance in the AR coefficient from
#                                                         their mean
#           params[3] (int) - batchSize: size of the batches of data we want to generate
#           params[4] (int) - sequenceLength: length of the sequence of data to generate for an
#                             element of a batch
#       seed (int) {default=a random number} - the seed of the random number generator
#                                              for this AR process
#   Outputs: (x, z)
#       x (tensor [batchSize x 4  x simuLength]) - a tensor of the real state values of the AR
#                               process separated into batch elements in the 1st dimension,
#                               complex and real state values of the AR process in the 2nd
#                               dimension, and separated by batch in the 3rd dimension
#           x[:,0,:] (float) - real value of the current actual state
#           x[:,1,:] (float) - real value of the next actual state
#           x[:,2,:] (float) - imaginary value of the current actual state
#           x[:,3,:] (float) - imaginary value of the next actual state
#
#       z (tensor [batchSize x 2 x sequenceLength x simuLength]) - a tensor of the measured state
#                               values of the AR process separated into batch elements in the 1st
#                               dimension, separated into real and complex numbers in the 2nd
#                               dimension, separated into a sequence of observations in the 3rd
#                               dimension, and separated by batch in the 4th dimension
#           z[:,0,:,:] (float) - real values of the measured state
#           z[:,1,:,:] (float) - imaginary values of the measured state

# Vocabulary definitions:
#   Sequence: A set of data generated from the same AR process. Has correlation between elements
#   Batch: A set of data that will be processed in parallel by the model.
#          Each element is a sequence
#   Series: A set of data generated to be processed once by the model. Each element is a batch.
#           This code generates one Series of length: simLength
def ARDatagenMismatch(params, seed=int(np.absolute(np.floor(100*np.random.randn())))):
    # Set the seed value for repeatable results
    simLength = params[0]
    AR_n = params[1]
    AR_coeffecient_noise_var = params[2]
    batchSize = params[3]
    sequenceLength = params[4]
    np.random.seed(seed)

    # Gain matrix on the previous values
    # TODO: Make the AR coefficient means be a parameter you can pass to the function
    arCoeffMeans = [0.5, 0.4]

    # Noise covariance matrix / noise mean
    Q = np.array([[0.1, 0], [0, 0.00000000001]])
    systNoiseMean = 0

    # Observation covariance matrix/ noise mean
    R = np.array([0.1])
    observNoiseMean=0

    # Pre-allocating the matrix that will store the true values of the predicted and current state
    x = np.zeros((batchSize, 4, simLength), dtype=float)
    # Pre-allocating the matrix that will store the measured data to be fed into the model
    z = np.zeros((batchSize, 2, sequenceLength, simLength), dtype=float)

    ### Loop for generating all the batches of data (a series) ###
    for i in range(0,simLength):
        ## Loop for generating a batch of data ##
        # Iterating through one additional time so that we can get the actual next state
        # and the current actual state
        for j in range(0, batchSize):
            F = ARCoeffecientGeneration(arCoeffMeans, AR_coeffecient_noise_var)
            # Loop for generating the sequence of data for each batch element #
            for m in range(0, sequenceLength + 1):
                # Generate system noise vector
                rSysNoise = np.divide(np.matmul(LA.cholesky(Q),
                                        np.random.randn(AR_n, 1) + systNoiseMean), np.sqrt(2))
                iSysNoise = np.divide(np.matmul(1j * LA.cholesky(Q),
                                        np.random.randn(AR_n, 1) + systNoiseMean), np.sqrt(2))
                v = rSysNoise + iSysNoise

                # Generate observation noise vector
                rObsNoise = np.divide(np.matmul(np.sqrt(R),
                                        np.random.randn(1) + observNoiseMean),np.sqrt(2))
                iObsNoise = np.divide(np.matmul(1j*np.sqrt(R),
                                        np.random.randn(1) + observNoiseMean),np.sqrt(2))
                w = rObsNoise + iObsNoise
                if(m==0):
                    x_complex = np.zeros((2), dtype=complex)
                    z_complex = 0 + 0j
                else:
                    # Don't think too hard about these lines
                    x_complex = np.matmul(F,x_complex)
                    x_complex[0] = (x_complex[0] + v)[0]
                    z_complex = x_complex[0] + w
                # Still in the measurement generation process
                if(m<sequenceLength):
                    # Storing the measured data in its appropriate batch element, its appropriate
                    # complex and real components, its appropriate sequence element, and the right
                    # series element
                    z[j,0,m,i] = z_complex.real
                    z[j,1,m,i] = z_complex.imag
                # If we are on the sequenceLength + 1 iteration we need to grab the current
                # true state (will be the next predicted true state from the measurements),
                # and the previous actual state (will be the current true state from the
                # measurements)
                else:
                    x[j,0,i] = x_complex[0].real
                    x[j,1,i] = x_complex[1].real
                    x[j,2,i] = x_complex[0].imag
                    x[j,2,i] = x_complex[1].imag
                # End of sequence generation loop
            # End of batch generation loop
        # End of series generation loop

    ##### Storing the data #####
    storageFilePath = './data'
    dataFile = 'data'
    logContent = {}
    logContent[u'measuredData'] = z
    logContent[u'predAndCurState'] = x
    matSave(storageFilePath,dataFile,logContent)

    # Return data
    return(x, z)