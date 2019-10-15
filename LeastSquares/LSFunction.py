import numpy as np

def LSTraining(trainData):
    measuredTrainData = trainData[1]
    trueStateTrainData = trainData[0]

    # N - length of the history/filter taps
    N = measuredTrainData.shape[1]  # sequenceLength
    # M - length of observation vector/number of LS equations
    M = measuredTrainData.shape[2]  # seriesLength
    # Pre-allocating the matrix that will store the measured data
    z = np.zeros((M, N), dtype=complex)

    # Pre-allocating the vector that will store the estimations/predictions
    x_est = np.zeros((M, 1), dtype=complex)
    x_pred = np.zeros((M, 1), dtype=complex)

    # Turn observations into complex form from separated real and imaginary components
    measuredStateDataComplex = measuredTrainData[0, :, :] + (measuredTrainData[1, :, :] * 1j)
    measuredStateDataComplex = np.squeeze(measuredStateDataComplex)

    # Turn real state values into complex form from separated real and imaginary components
    ARValuesComplex = trueStateTrainData[0, :, :] + (trueStateTrainData[1, :, :] * 1j)
    ARValuesComplex = np.squeeze(ARValuesComplex)

    # Construct matrices for MSE training
    z[:, :] = np.transpose(np.flipud(measuredStateDataComplex[0:N, :]))
    x_est[:, 0] = ARValuesComplex[N - 1, :]
    x_pred[:, 0] = ARValuesComplex[N, :]

    # Removing the first elements of the data (last row after flipping) because it
    # is all zeros and that causes problems with matrix inverses
    # z = z[:, 0:-1]

    z_psuedoInverse = np.linalg.pinv(z)

    # estimate coefficients
    lsEstimateCoeffs = np.matmul(z_psuedoInverse, x_est)

    # prediction coefficients
    lsPredictionCoeffs = np.matmul(z_psuedoInverse, x_pred)

    return [lsEstimateCoeffs, lsPredictionCoeffs]

def LSTesting(estAndPredFilterCoeffs, testData):

    # Expecting the first input to be a list or tuple formatted exactly like LSTrainings output
    a_ls = estAndPredFilterCoeffs[0]
    b_ls = estAndPredFilterCoeffs[1]

    # Loading the necessary data from the testData dictionary
    measuredStateData = testData[1]
    ARValues = testData[0]
    # N - length of the history/filter taps
    N = measuredStateData.shape[1]  # sequenceLength
    # M - length of observation vector/number of LS equations
    M = measuredStateData.shape[2]  # seriesLength
    # Pre-allocating the matrix that will store the measured data
    z = np.zeros((M, N), dtype=complex)

    # Pre-allocating the vector that will store the estimations/predictions
    x_est = np.zeros((M, 1), dtype=complex)
    x_pred = np.zeros((M, 1), dtype=complex)

    # Turn observations into complex form from separated real and imaginary components
    measuredStateDataComplex = measuredStateData[0, :, :] + (measuredStateData[1, :, :] * 1j)
    measuredStateDataComplex = np.squeeze(measuredStateDataComplex)

    # Turn real state values into complex form from separated real and imaginary components
    ARValuesComplex = ARValues[0, :, :] + (ARValues[1, :, :] * 1j)
    ARValuesComplex = np.squeeze(ARValuesComplex)

    # Construct matrices for MSE training
    z[:, :] = np.transpose(np.flipud(measuredStateDataComplex[0:N, :]))
    x_est[:, 0] = ARValuesComplex[N - 1, :]
    x_pred[:, 0] = ARValuesComplex[N, :]

    # Removing the first elements of the data (last row after flipping) because it
    # is all zeros and that causes problems with matrix inverses
    # z = z[:, 0:-1]

    # Calculate MSE of estimation
    f = abs((x_est - np.matmul(z, a_ls))) ** 2
    MSEE = np.mean(f)

    # Calculate MSE of prediction
    f = abs((x_pred - np.matmul(z, b_ls))) ** 2
    MSEP = np.mean(f)

    # Returns the Mean Squared Estimated error, and the Meas Squared Predicted error in that order
    return(MSEE, MSEP)
