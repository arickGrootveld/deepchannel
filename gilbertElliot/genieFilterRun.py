import argparse as ap
from utilities import matSave
import hdf5storage as hdf5
import numpy as np

def genieKFTesting(testData, ARCoeffs, debug=False, initTest=False, **kwargs): 
    # Preset Parameters, because we are on a strict timeline
    measuredStateData = testData[1]
    trueStateData = testData[2]
    AR_n = 2

    
    measuredStateDataTest = np.empty((2, measuredStateData.shape[1] + 
        measuredStateData.shape[2]))
    
    # Variables for deciding length of variables
    seriesLength = trueStateData.shape[1]
    sequenceLength = measuredStateData.shape[1]


    # Looping through data to grab the relevant samples
    for p in range(0, seriesLength):
        if p == 0:
            measuredStateDataTest[:, 0:sequenceLength] = measuredStateData[:, :, 0]
        else:
            measuredStateDataTest[:, sequenceLength - 1 + p] = measuredStateData[:, sequenceLength - 1, p]

    # Throwing out samples that we don't need if we are
    # testing the initializations
    # Note that this is built assuming testInit was passed to
    # the data gen script that generated the data being used
    if initTest:
        seqLen = measuredStateDataTest.shape[1] - trueStateData.shape[1]
        inter1 = np.zeros([2, trueStateData.shape[1]-1])
        
        inter1[0,0] = 0
        inter1[1,0] = 0
        inter1[:,1:] = measuredStateDataTest[:, 1+sequenceLength:measuredStateDataTest.shape[1]-1]
        measuredStateDataTest = inter1
        seriesLength = seriesLength - sequenceLength

        # TODO: Figure out how we have to change ARCoeffs for this to work
        # TODO: properly
        inter1 = ARCoeffs[:, sequenceLength:ARCoeffs.shape[1] - 1]
        ARCoeffs = inter1

    ##### Kalman Filter Implementation #####
    # Initializing the Kalman Filter variables
    # For a better understanding of the vocabulary used in this code, please consult the mismatch_data_gen
    # file, above the ARDatagenMismatch function

    # Prediction/estimate formatted into current and last state value in the 1st dimension,
    # by sequence in the 2nd dimension, and by series in the 3rd dimension
    x_correction = np.zeros((AR_n, 
                             seriesLength+sequenceLength - 1), dtype=complex)
    x_prediction = np.zeros((AR_n, 
                             seriesLength+sequenceLength - 1), dtype=complex)

    # Other Kalman Filter variables formatted into a matrix/vector of values in the 1st and 2nd dimensions,
    # by sequence in the 3rd dimension, and by series in the 4th dimension
    kalmanGain = np.zeros((AR_n, 1, 
                           seriesLength+sequenceLength - 1))
    minPredMSE = np.zeros((AR_n, AR_n, 
                           seriesLength+sequenceLength - 1))
    minMSE = np.zeros((AR_n, AR_n, 
                       seriesLength+ sequenceLength - 1))

    # Initializing the correction value to be the expected value of the starting state
    x_correction[:, 0] = np.array([0,0])
    x_prediction[:, 0] = np.array([0, 0])
    # Initializing the MSE to be the variance in the starting value of the sequence
    minMSE[:, :, 0] = np.array([[1, 0], [0, 1]])

    ## Kalman Filter parameters to be used

    # Covariance of the process noise
    Q = np.array([[0.1, 0], [0, 0]])

    # Observation Matrix mapping the observations into the state space
    H = np.array([1, 0])[np.newaxis]

    # Covariance of the measurements
    R = np.array([0.1])

    # Initializing the total MSE that we will use to follow the MSE through each iteration
    totalTruePredMSE = 0
    totalTrueEstimateMSE = 0
    
    if(debug):
        if(initTest):
            instaErrs = np.empty([1, seriesLength + sequenceLength - 1])
            kfPreds = np.empty([1, seriesLength + sequenceLength - 1], dtype=np.complex128)
        else:
            instaErrs = np.empty([1, seriesLength])
            kfPreds = np.empty([1, seriesLength], dtype=np.complex128)

    for i in range(1, seriesLength + sequenceLength - 1):

        # Defining F to be the coefficients of the current sample
        F = np.array([ARCoeffs[:, i],[1, 0]])

        # Loop through a sequence of data
        #############################################################################
        ############################# KALMAN FILTER 1  ##############################
        # This is the original Kalman Filter - does not know the actual AR coeffecients of the
        # AR processes that are input, only knows the theoretical mean values.
        # Formatting the measured data properly
        measuredDataComplex = measuredStateDataTest[0, i] + (measuredStateDataTest[1, i] * 1j)
        
        # Calculating the prediction of the next state based on the previous estimate
        x_prediction[:, i] = np.matmul(F, x_correction[:, i-1])
        # Calculating the predicted MSE from the current MSE, the AR Coefficients,
        # and the covariance matrix
        minPredMSE[:, :, i] = np.matmul(np.matmul(F, minMSE[:, :, i-1]), np.transpose(F)) + Q
        # Calculating the new Kalman gain
        intermediate1 = np.matmul(minPredMSE[:, :, i], np.transpose(H))
        # Intermediate2 should be a single dimensional number, so we can simply just divide by it
        intermediate2 = np.linalg.inv(R + np.matmul(np.matmul(H, minPredMSE[:, :, i]),
                                                        np.transpose(H)))
        kalmanGain[:, :, i] = np.matmul(intermediate1, intermediate2)

        # Calculating the State Correction Value
        intermediate1 = np.matmul(H, x_prediction[:, i])
        intermediate2 = measuredDataComplex - intermediate1
        x_correction[:, i] = x_prediction[:, i] + np.matmul(kalmanGain[:, :, i],
                                                                      intermediate2)
        # Calculating the MSE of our current state estimate
        intermediate1 = np.identity(AR_n) - np.matmul(kalmanGain[:, :, i], H)
        minMSE[:, :, i] = np.matmul(intermediate1, minPredMSE[:, :, i])

        ############################# KALMAN FILTER 1  ##############################

                
        # Updating the correction value to persist between rows, as the Kalman Filter
        # is an iterative approach, not having this persistence causes it to take 
        # longer to converge to the Riccati equation than it should
        if((i >= sequenceLength - 1) and not initTest):

            ## Calculating the actual MSE between the kalman filters final prediction, and the actual value ##
            # Converting the true states into their complex equivalents
            currentTrueStateComplex = trueStateData[0, i-sequenceLength + 1] + (1j * trueStateData[2, i-sequenceLength + 1])
            
            finalPrediction = x_prediction[:,i][0]
            finalEstimate = x_correction[:, i][0]
            
            # Calculating the instantaneous MSE of our estimate and prediction
            trueEstimateMSE = np.absolute(finalEstimate - currentTrueStateComplex) ** 2
            
            # Prediction was made before this state had been fed into KF, so
            # this comparison is fine. This current prediction was what the KF
            # thought this current state would be
            truePredictionMSE = np.absolute(finalPrediction - currentTrueStateComplex) ** 2

            if debug:
                instaErrs[0, i-sequenceLength + 1] = truePredictionMSE
                kfPreds[0, i-sequenceLength + 1] = finalPrediction

            totalTrueEstimateMSE += trueEstimateMSE
            totalTruePredMSE += truePredictionMSE

        # Recording all the predictions from the beggining and later
        elif initTest:
            currentTrueStateComplex = trueStateData[0, i+1] + (1j * trueStateData[2, i+1])

            finalPrediction = x_prediction[:, i][0]
            finalEstimate = x_correction[:, i][0]

            # Calculating the instantaneous MSE of our estimate and prediction
            trueEstimateMSE = np.absolute(finalEstimate - currentTrueStateComplex) ** 2
            truePredictionMSE = np.absolute(finalPrediction - currentTrueStateComplex) ** 2

            if debug:
                instaErrs[0, i] = truePredictionMSE
                kfPreds[0, i] = finalPrediction

            totalTrueEstimateMSE += trueEstimateMSE
            totalTruePredMSE += truePredictionMSE


    totalTrueEstimateMSE = totalTrueEstimateMSE / (seriesLength)
    totalTruePredMSE = totalTruePredMSE / (seriesLength)
    

    # Different return pattern if we are in debug or not
    if debug:
        return (totalTrueEstimateMSE, totalTruePredMSE, instaErrs, kfPreds)
    else:
        return (totalTrueEstimateMSE, totalTruePredMSE)





parser = ap.ArgumentParser(description='Script to test performance of Genie Kalman Filter')

parser.add_argument('--testFile', type=str, default='None',
                    help='file containing test data to check performance on'
                    '(default: None)')
parser.add_argument('--initTest', action='store_true', 
                    help="Whether to perform init test functionality"
                    "(default: False)")


args = parser.parse_args()

testFile = args.testFile

testData = hdf5.loadmat(testFile)

testDataLen = len(testData['LSandKFTestData'])

gkfSaveData = dict()
gkfRuns = []

for i in range(0, testDataLen):
    if i == 0:
        # Loading data from the mode switching problem
        data = testData['LSandKFTestData'][i]
        coeffs = testData['testDataInfo'][i]['channelCoefficients']

        genieRun = genieKFTesting(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in mode switching: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs
        
        gkfRuns.append(gkfRunInfo)
    
    if i == 1:
        # Testing the genie KF performance on the good state dataset
        data = testData['LSandKFTestData'][i]
        coeffs = testData['testDataInfo'][i]['channelCoefficients']

        genieRun = genieKFTesting(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in only Good States: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs

        gkfRuns.append(gkfRunInfo)

    if i == 2:
        # Testing the genie KF performance on the bad state dataset
        data = testData['LSandKFTestData'][2]
        coeffs = testData['testDataInfo'][2]['channelCoefficients']

        genieRun = genieKFTesting(data, coeffs, debug=True, initTest=args.initTest)

        predMSEVal = genieRun[1]

        print("Predicted MSE of Genie KF in only Bad States: ", predMSEVal)

        genKFInstaErrs = genieRun[2]

        gkfRunInfo = dict()

        gkfRunInfo[u'genKFInstaErrs'] = genKFInstaErrs

        gkfRuns.append(gkfRunInfo)

gkfSaveData['gkfRuns'] = gkfRuns
gkfSaveData['file'] = testFile
gkfSaveData['initTest'] = args.initTest

matSave('logs', 'genieLog', gkfSaveData)
