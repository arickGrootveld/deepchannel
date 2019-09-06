import numpy as np

def KFTesting(testData, ARCoeffs):

    # Preset Parameters, because we are on a strict timeline
    measuredStateData = testData[1]
    trueStateData = testData[2]
    AR_n = 2


    # Variables for deciding length of variables
    sequenceLength = measuredStateData.shape[1]
    seriesLength = measuredStateData.shape[2]

    ##### Kalman Filter Implementation #####
    # Initializing the Kalman Filter variables
    # For a better understanding of the vocabulary used in this code, please consult the mismatch_data_gen
    # file, above the ARDatagenMismatch function

    # Prediction/estimate formatted into current and last state value in the 1st dimension,
    # by sequence in the 2nd dimension, and by series in the 3rd dimension
    x_correction = np.zeros((AR_n, sequenceLength,
                             seriesLength), dtype=complex)
    x_prediction = np.zeros((AR_n, sequenceLength,
                             seriesLength), dtype=complex)

    # Other Kalman Filter variables formatted into a matrix/vector of values in the 1st and 2nd dimensions,
    # by sequence in the 3rd dimension, and by series in the 4th dimension
    kalmanGain = np.zeros((AR_n, 1, sequenceLength,
                           seriesLength))
    minPredMSE = np.zeros((AR_n, AR_n, sequenceLength,
                           seriesLength))
    minMSE = np.zeros((AR_n, AR_n, sequenceLength,
                       seriesLength))

    # Initializing the minPredMSE to be 1, because the AR Process starts in a random state with unit variance
    # and 0 mean
    minPredMSE[0,0, 0, :] = np.identity(2)

    # Initializing the correction value to be the expected value of the starting state
    x_correction[:, 0, 0] = np.array([0, 0])
    # Initializing the MSE to be the variance in the starting value of the sequence
    ## This is just making the first matrix corresponding to the MMSE of each sequence be an identity ##
    ## matrix ##
    minMSE[0, 0, 0, :] = 1
    minMSE[1, 1, 0, :] = 1

    ## Kalman Filter parameters to be used

    # F matrix is made up of AR process mean values
    F = np.array([ARCoeffs, [1, 0]])

    # Covariance of the process noise
    Q = np.array([[0.1, 0], [0, 0]])

    # Observation Matrix mapping the observations into the state space
    H = np.array([1, 0])[np.newaxis]

    # Covariance of the measurements
    R = np.array([0.1])

    # Initializing the total MSE that we will use to follow the MSE through each iteration
    totalTruePredMSE = 0
    finalTruePredMSE = 0
    totalTrueEstimateMSE = 0
    finalTrueEstimateMSE = 0

    for i in range(0, seriesLength):
        # Loop through a sequence of data
        for q in range(1, sequenceLength):
            #############################################################################
            ############################# KALMAN FILTER 1  ##############################
            # This is the original Kalman Filter - does not know the actual AR coeffecients of the
            # AR processes that are input, only knows the theoretical mean values.

            # Formatting the measured data properly
            measuredDataComplex = measuredStateData[0, q, i] + (measuredStateData[1, q, i] * 1j)

            # Calculating the prediction of the next state based on the previous estimate
            x_prediction[:, q, i] = np.matmul(F, x_correction[:, q - 1, i])

            # Calculating the predicted MSE from the current MSE, the AR Coefficients,
            # and the covariance matrix
            minPredMSE[:, :, q, i] = np.matmul(np.matmul(F, minMSE[:, :, q - 1, i]), np.transpose(F)) + Q

            # Calculating the new Kalman gain
            intermediate1 = np.matmul(minPredMSE[:, :, q, i], np.transpose(H))
            # Intermediate2 should be a single dimensional number, so we can simply just divide by it
            intermediate2 = np.linalg.inv(R + np.matmul(np.matmul(H, minPredMSE[:, :, q, i]),
                                                        np.transpose(H)))
            kalmanGain[:, :, q, i] = np.matmul(intermediate1, intermediate2)

            # Calculating the State Correction Value
            intermediate1 = np.matmul(H, x_prediction[:, q, i])
            intermediate2 = measuredDataComplex - intermediate1
            x_correction[:, q, i] = x_prediction[:, q, i] + np.matmul(kalmanGain[:, :, q, i],
                                                                      intermediate2)
            # Calculating the MSE of our current state estimate
            intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain[:, :, q, i], H)
            minMSE[:, :, q, i] = np.matmul(intermediate1, minPredMSE[:, :, q, i])

        ############################# KALMAN FILTER 1  ##############################

        ## Calculating the actual MSE between the kalman filters final prediction, and the actual value ##
        # Converting the true states into their complex equivalents
        currentTrueStateComplex = trueStateData[0, i] + (1j * trueStateData[2, i])
        nextTrueStateComplex = trueStateData[1, i] + (1j * trueStateData[3, i])

        finalPrediction = np.matmul(F, x_correction[:, q, i])[0]
        finaEstimate = x_correction[:, q, i][0]

        # Calculating the instantaneous MSE of our estimate and prediction
        trueEstimateMSE = np.absolute(finaEstimate - currentTrueStateComplex) ** 2
        truePredictionMSE = np.absolute(finalPrediction - nextTrueStateComplex) ** 2

        totalTrueEstimateMSE += trueEstimateMSE
        totalTruePredMSE += truePredictionMSE

    totalTrueEstimateMSE = totalTrueEstimateMSE / (seriesLength)
    totalTruePredMSE = totalTruePredMSE / (seriesLength)

    return (totalTrueEstimateMSE, totalTruePredMSE)

