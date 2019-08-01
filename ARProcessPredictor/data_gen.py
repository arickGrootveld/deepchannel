# Generate an AR process - for dummy data

import numpy as np
import os
import math
import os.path as path

def ARdatagen(AR_n, simLength, seed, polar=False):
    # Set the seed value for repeatable results
    np.random.seed(seed)

    # Gain matrix on the previous values
    F = np.array([[0.5, 0.4], [1, 0]])

    # Noise covariance matrix / noise mean
    Q = np.array([[0.1, 0], [0, 0.00000000001]])
    systNoiseMean = 0

    # Observation covariance matrix/ noise mean
    R = np.array([0.1])
    observNoiseMean=0

    # Generate system noise vector storage
    v = np.zeros((AR_n, 1, simLength), dtype=complex)

    # Generate observation noise vector storage
    w = np.zeros(simLength, dtype=complex)

    # Generate channel state vector storage
    x = np.zeros((AR_n, 1, simLength), dtype=complex)

    # Generate observation vector storage
    z = np.zeros(simLength, dtype=complex)

    # Processing loop for data generation
    for i in range(1,simLength) :

        # Generate noise vector
        arg1 = np.matmul(np.linalg.cholesky(Q), np.random.randn(AR_n, 1))
        arg1 = np.divide(arg1, np.sqrt(2))
        arg2 = np.matmul(1j * np.linalg.cholesky(Q), np.random.randn(AR_n, 1))
        arg2 = np.divide(arg2, np.sqrt(2))
        v[:, :, i] = systNoiseMean + arg1 + arg2
        v[1, :, i] = 0 # Explicit check for data

        # Create AR process data - starting data at zero (first value = 0)
        x[:, :, i] = np.matmul(F, x[:, :, i-1]) + v[:, :, i]

        # Generate observation noise (z estimations of state)
        arg1 = np.matmul(np.sqrt(R), np.random.randn(1))
        arg1 = np.divide(arg1, np.sqrt(2))
        arg2 = np.matmul(1j * np.sqrt(R), np.random.randn(1))
        arg2 = np.divide(arg2, np.sqrt(2))
        w[i] = observNoiseMean + arg1 + arg2

        # Generate the observation vector
        z[i] = x[0, :, i][0] + w[i]

    # Resize hk to be a 2D matrix
    x = x[0, :]

    # Change the data format
    x = x[:].view(float).reshape(-1, 2)
    z = z[:].view(float).reshape(-1, 2)


    # !!!!!!!!!!!!! TESTING POLAR FORM - REMOVE/COMMENT OUT TO CHANGE BACK !!!!!!!!!!!!!!!!
    if(polar):
        for i in range(0, simLength - 1):

            tempreal =  x[i, 0]
            tempimag =  x[i, 1]

            tempcomplex = np.complex(tempreal, tempimag)

            x[i, 0] = np.absolute(tempcomplex)
            x[i, 1] = np.angle(tempcomplex)

            tempreal = z[i, 0]
            tempimag = z[i, 1]

            tempcomplex = np.complex(tempreal, tempimag)

            z[i, 0] = np.absolute(tempcomplex)
            z[i, 1] = np.angle(tempcomplex)


    # Set up data directory
    fileSpaceFound = False
    logNumber = 0
    logFileData = []
    if not (path.exists('./data')):
        os.mkdir('./data', 0o755)

    # Create data files
    while (fileSpaceFound == False):
        logPath = './data/trueStateData' + str(logNumber) + '.txt'
        observedLogPath = './data/observedData' + str(logNumber) + '.txt'
        if not (path.exists(logPath) or path.exists(observedLogPath)):
            fileSpaceFound = True

            # Real data saved in left column, imaginary portion saved in right column - Save data to text files
            np.savetxt(logPath, x, delimiter=',')
            np.savetxt(observedLogPath, z, delimiter=',')

        logNumber += 1

    # Return data
    return(x, z)
