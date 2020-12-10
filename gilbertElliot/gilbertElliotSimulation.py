# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#            TCN Based Channel Estimation Neural Network Training
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import torch
import argparse
from argparse import RawTextHelpFormatter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import os
from os import path
import numpy as np

# Imports for utility code
import hdf5storage as hdf5s
import os
import os.path as path

import sys

# from utilities import convertToBatched, matSave, shuffleMeasTrainData, shuffleTrueTrainData
# from LeastSquares.LSFunction import LSTesting, LSTraining
# from KalmanFilter.KFFunction import KFTesting2

# Timer for logging how long the training takes to execute
import time
start = time.time()

# Importing file handler for .mat files
import hdf5storage as hdf5s


### Defining internal functions, some borrowed from http://github.com/locuslab/TCN
### some created during the writting of the paper this was developed for

## Code borrowed from Locus Labs
# from torch import nn

# Define the model
class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:,:,-1])


# This python file defines the tcn (temporal convolution neural network)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


############################### Utility code written for general purposes ###########################################

# matSave: Function that saves data to a specified .mat file, with the specific file it will be
#          saved to being 'directory/basename{#}.mat', where # is the lowest number that will
#          not save over another file with the same basename
#   Inputs: (directory, basename, data)
#       directory (str) - the directory for the data to be saved in
#       basename (str) - the name of the file you want the data saved to before the appended number
#       data (dict) - a dict of data to be saved to the .mat file
#   Outputs: (logName)
#       logName (str) - the name of the file that the data was saved to
def matSave(directory, basename, data):
    # Create the data directory if it doesn't exist
    if not (path.exists(directory)):
        os.mkdir(directory, 0o755)
    fileSpaceFound = False
    logNumber = 0
    # Creating the log file in a new .mat file that doesn't already exist
    while (not fileSpaceFound):
        logNumber += 1
        logName = directory + '/' + basename + str(logNumber) + '.mat'
        if not (path.exists(logName)):
            print('data saved to: ', logName)
            fileSpaceFound = True
    # Saving the data to the log file
    hdf5s.savemat(logName, data)
    return(logName)

def convertToBatched(systemDataToBeConverted, observedDataToBeFormatted, batchSize):
    numSequences = observedDataToBeFormatted.shape[2]
    seqLength = observedDataToBeFormatted.shape[1]
    seriesLength = int(numSequences/batchSize)

    trueState = np.empty((batchSize, 4, seriesLength), dtype=float)
    measuredState = np.empty((batchSize, 2, seqLength, seriesLength), dtype=float)

    for i in range(seriesLength):
        trueState[:, :, i] = np.transpose(systemDataToBeConverted[:, i * batchSize:(i + 1) * batchSize])
        measuredState[:, :, :, i] = np.swapaxes(
            np.transpose(observedDataToBeFormatted[:, :, i * batchSize:(1 + i) * batchSize]), 1, 2)
    return(trueState, measuredState)

# TODO: Make this and the following function more general, so they could
# TODO: be used for shuffling matrices of any shape and dimensionality
def shuffleMeasTrainData(dataToBeShuffled):
    import torch as t
    dataDims = dataToBeShuffled.shape
    shuffledData = t.zeros(dataDims)

    totalDims = dataDims[0] * dataDims[3]
    
    randomPermutation = t.randperm(totalDims)
    cnt = -1
    for m in range(0, dataDims[0]):
        for n in range(0, dataDims[3]):
            cnt = cnt + 1
            i = randomPermutation[cnt] % dataDims[0]
            j = randomPermutation[cnt] // dataDims[0]

            shuffledData[m, :, :, n] = dataToBeShuffled[i, :, :, j]

    return (shuffledData, randomPermutation)


def shuffleTrueTrainData(dataToBeShuffled, randomPermutation):
    import torch as t
    dataDims = dataToBeShuffled.shape
    shuffledData = t.zeros(dataDims)
    cnt = -1
    for m in range(0, dataDims[0]):
        for n in range(0, dataDims[2]):
            cnt = cnt + 1
            i = randomPermutation[cnt] % dataDims[0]
            j = randomPermutation[cnt] // dataDims[0]

            shuffledData[m, :, n] = dataToBeShuffled[i, :, j]

    return shuffledData




############################### Code used to simulate alternative methods of for sequence modeling ###########################################
# Redone Kalman Filter function to make it faster and make more sense
# Now flattens the data and doesn't do the whole double looping thing anymore
# Has slightly more accuracy for some reason, not sure about this
def KFTesting(testData, ARCoeffs, debug=False, initTest=False, **kwargs): 
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
    totalTrueEstimateMSE = 0
    
    if(debug):
        if(initTest):
            instaErrs = np.empty([1, seriesLength + sequenceLength - 1])
            kfPreds = np.empty([1, seriesLength + sequenceLength - 1], dtype=np.complex128)
        else:
            instaErrs = np.empty([1, seriesLength])
            kfPreds = np.empty([1, seriesLength], dtype=np.complex128)

    for i in range(1, seriesLength + sequenceLength - 1):
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

    z_psuedoInverse = np.linalg.pinv(z)

    # estimate coefficients
    lsEstimateCoeffs = np.matmul(z_psuedoInverse, x_est)

    # prediction coefficients
    lsPredictionCoeffs = np.matmul(z_psuedoInverse, x_pred)

    return [lsEstimateCoeffs, lsPredictionCoeffs]

def LSTesting(estAndPredFilterCoeffs, testData, debug=False, **kwargs):
    
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

    # Calculate MSE of estimation
    f = abs((x_est - np.matmul(z, a_ls))) ** 2
    MSEE = np.mean(f)

    # Calculate MSE of prediction
    f = abs((x_pred - np.matmul(z, b_ls))) ** 2
    MSEP = np.mean(f)

    if debug:
        lsPred = np.matmul(z, b_ls)
        return(MSEE, MSEP, lsPred, f)
    else:
        # Returns the Mean Squared Estimated error, and the Meas Squared Predicted error in that order
        return(MSEE, MSEP)


############################### Beggining of code used for model generation and training ###########################################

### ~~~~~~~~~~~~~~~~~~~~ ARGUMENT PARSER ~~~~~~~~~~~~~~~~~~~~ ###
### Argument parser for easier and cleaner user interface

# Create argument parser
parser = argparse.ArgumentParser(description='Sequence Modeling - Complex Channel Gain Estimation', formatter_class=RawTextHelpFormatter)

# Batch size
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')

# CUDA enable
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')

parser.add_argument('--cuda_device', type=str, default='all',
                    help='the cuda device to be used. If all is ' +
                    'specified then it will use every GPU available.' +
                    '(default: all)')

# Dropout rate
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0)')

# Gradient clipping
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')

# Training epochs
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit (default: 1)')

# Kernel (filter) size
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')

# Number of levels in the model
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')

# Reporting interval
parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='report interval (default: 2000')

# Learning rate
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate (default: 2e-4)')

# Optimizer
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')

# Hidden units per layer
parser.add_argument('--nhid', type=int, default=3,
                    help='number of hidden units per layer (default: 3)')

# Random seed value
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

# Load data to train with from data file
parser.add_argument('--trainDataFile', type=str, default='None',
                    help='file path to load training data from, if None then it will generate its own data '
                         '(default=\'None\'){if a data file is loaded from, the --simLength, --batch_size, '
                         '--seq_len, --AR_var, and --seed parameters will do nothing, because these values are used for data '
                         'generation. The same applies to the --testDataFile argument and --evalDataFile argument} \r\n')

# Load data to evaluate models with from data file
parser.add_argument('--evalDataFile', type=str, default='None',
                    help='file path to load eval data from, if None then it will generate its own data (default=\'None\')')

# Load data to test TCN, LS, and KF
parser.add_argument('--testDataFile', type=str, default='None',
                    help='file path to load test data from, if None then it will generate its own data (default=\'None\')')


# Location to load the model from
parser.add_argument('--model_path', type=str, default='None',
                    help='The location to load the model parameters from. If set to None, will generate new model'
                         'from training and evaluation data, otherwise it will skip the training and evaluation loops'
                         'and simply test the model loaded (default=\'None\')')
# If model is loaded from a path, will skip over the training and evaluation loops and go straight to testing. This will
# ignore all data generation specified for train and eval, and will only generate/load data for the testing process.

# Setting the model and setup to debug mode so that we can recover instantaneous squared errors, etc.
parser.add_argument('--debug', action='store_true',
                    help='set code to debug mode (default: False)')

# Coefficients used by the Kalman filter for the F matrix it assumes the Gauss Markov Process uses
parser.add_argument('--KFCoeffs', nargs='+', default=[0.3, 0.1],
                    help='Coefficients Passed to the Kalman Filter, will depend on the scenario you are looking at'
                         '(default: [0.3, 0.1]')

parser.add_argument('--initTest', action='store_true',
                    help='perform various operations to test how'
                    'all the tested methods perform at channel initialization'
                    '(default: false)')

parser.add_argument('--bias_removal', action='store_true',
                    help='do data preprocessing to make each'
                    'sequence start from the origin. Meant for '
                    'Maneuvering Targets Model System'
                    '(default: fals)')


# Parse out the input arguments
args = parser.parse_args()

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~ Make runs reproducable ~~~~~~~~~~~~~~~~~ ###

# This sets the seed value for random number generators, making
# run results repeatable
torch.manual_seed(args.seed)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~ Logging Verification ~~~~~~~~~~~~~~~~~ ###

# Set up directory creation
fileSpaceFound = False
logNumber=0

# Create the directory if non-existent
if not (path.exists('./logs')):
        os.mkdir('./logs', 0o755)

# Creating a new log file
while(fileSpaceFound==False):
    logNumber+=1
    logName='./logs/log'+str(logNumber)+'.mat'
    if not (path.exists(logName)):
        print(logName)
        fileSpaceFound=True

# Logging the parameters of the model
print('model parameters: ', args)
fileContent = {}

# Writing to the file here even though it will get overriden, because we want other TCN processes to see
# that data will already be stored in this file, and to increment past it
hdf5s.savemat(logName, fileContent)

# Saving the command line arguments that were passed as the model parameters
fileContent[u'model parameters'] = repr(args)
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ CUDA  CHECK ~~~~~~~~~~~~~~~~~~~~~~ ###

if torch.cuda.is_available():
    if not args.cuda:
        print("CUDA device detected. To run training with CUDA, input command argument --cuda")

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~ ###

# Real and Imaginary components (input and output)
input_channels = 2
n_classes = 2

batch_size = args.batch_size
epochs = args.epochs
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
lr = args.lr

seed = args.seed
optimMethod = args.optim

debug_mode = args.debug
biasRemoval = args.bias_removal

trainFile = args.trainDataFile
evalFile = args.evalDataFile
testFile = args.testDataFile

KFARCoeffs = []
for KFCoeff in args.KFCoeffs:
    KFARCoeffs.append(float(KFCoeff))


# Determining whether this run is for testing only, or if it is for training a model to be tested
testSession = False
if(args.model_path != 'None'):
    testSession = True
    if(args.cuda):
        modelContext = torch.load(args.model_path)
    else:
        device = torch.device('cpu')
        modelContext = torch.load(args.model_path, map_location=device)
modelPath = args.model_path
### ~~~~~~~~~~~~~~~ LOAD DATA/GENERATE MODEL ~~~~~~~~~~~~~~~~ ###
# Here we have the option of loading from a saved .mat file or just calling the data generation
# function explicitly

# AR data generation parameters
AR_n = 2

# ~~~~~~~~~~~~~~~~~~ LOAD TRAINING SET
if not testSession:
    # Parameter to determine how much data to train the Least Squares Filter on
    pseudoInverseDataThreshold = int(1e7)

    # No train file was specified, so we raise an exception, since a train file needs to be specified
    if(trainFile == "None"):
        # You can specify train files via the --trainDataFile parameter to the CLI
        # Train files are generally generated from one of the files provided in this repo, 
        # such as gilbertElliotDataGen.py
        raise Exception('Need to specify a train data file for this function to work')
    else:
        # Grab the data from the .mat file
        trainDataDict = hdf5s.loadmat(trainFile)
        print('train data loaded from: {}'.format(trainFile))
        # Convert the loaded data into batches for the TCN to run with
        trueStateTRAIN, measuredStateTRAIN = convertToBatched(trainDataDict['finalStateValues'], trainDataDict['observedStates'],
                                                              batch_size)

        fileContent[u'trainDataFile'] = trainFile
        fileContent[u'trainDataSize'] = trueStateTRAIN.shape

        # Setting the number of batches of train data to be what is supplied in the file
        trainSeriesLength = trueStateTRAIN.shape[2]
        trainDataLen = trainDataDict['finalStateValues'].shape[1]

        # TODO: Improve the format at some point in the future, but for now we are just grabbing the trainingData to train
        # TODO: the LS

        # Only grabbing a portion of the train samples for the LS training, because otherwise the pseudo-inverse
        # will use too much data
        if(trainDataLen >= pseudoInverseDataThreshold):
            LSTrainData = [trainDataDict['systemStates'][:,:,0:pseudoInverseDataThreshold],
                           trainDataDict['observedStates'][:,:,0:pseudoInverseDataThreshold]]
        else:
            LSTrainData = [trainDataDict['systemStates'], trainDataDict['observedStates']]


    # Convert numpy arrays to tensors
    trueStateTRAIN = torch.from_numpy(trueStateTRAIN)
    measuredStateTRAIN = torch.from_numpy(measuredStateTRAIN)

    # ~~~~~~~~~~~~~~~~~~ LOAD EVALUATION SET
    # No eval file was specified, so we raise an exception, since a eval file needs to be specified
    if(trainFile == "None"):
        # You can specify eval files via the --evalDataFile parameter to the CLI
        # Eval files are generally generated from one of the files provided in this repo, 
        # such as gilbertElliotDataGen.py
        raise Exception('Need to specify a train data file for this function to work')
    # loading the data from the file
    else:
        # Grab the data from the .mat file
        evalDataDict = hdf5s.loadmat(evalFile)
        print('eval data loaded from: {}'.format(evalFile))
        trueStateEVAL, measuredStateEVAL = convertToBatched(evalDataDict['finalStateValues'],
                                                            evalDataDict['observedStates'],
                                                            batch_size)
        fileContent[u'evalDataFile'] = evalFile
        fileContent[u'evalDataSize'] = trueStateEVAL.shape

        evalSeriesLength = trueStateEVAL.shape[2]
        evalDataLen = evalSeriesLength * trueStateEVAL.shape[0]
    # Convert numpy arrays to tensors
    trueStateEVAL = torch.from_numpy(trueStateEVAL)
    measuredStateEVAL = torch.from_numpy(measuredStateEVAL)


# ~~~~~~~~~~~~~~~~~~ LOAD TEST SET

# No test file was specified, so we raise an exception, since a test file needs to be specified
if(testFile == "None"):
    # You can specify test files via the --testDataFile parameter to the CLI
    # Test files are generally generated from one of the files provided in this repo, 
    # such as gilbertElliotDataGen.py using the --testDataGen cli argument
    raise Exception('Need to specify a test file for this function to work')
# Loading all data from a file
else:
    testDataToBeLoaded = hdf5s.loadmat(testFile)
    trueStateTEST = testDataToBeLoaded['trueStateTEST']
    measuredStateTEST = testDataToBeLoaded['measuredStateTEST']
    testDataInfo = testDataToBeLoaded['testDataInfo']
    LSandKFTestData = testDataToBeLoaded['LSandKFTestData']

    # overriding parameters from command line, because we are loading from file
    testSetLen = trueStateTEST.shape[0]
    testSeriesLength = trueStateTEST.shape[3]

    altAlgs = False
    if (('IMMPredVals' in testDataToBeLoaded) and
        ('GKFPredVals' in testDataToBeLoaded)):
        altAlgs = True
        IMMPredVals = testDataToBeLoaded['IMMPredVals']
        GKFPredVals = testDataToBeLoaded['GKFPredVals']

    print('test data loaded from: {}'.format(testFile))
    fileContent[u'testDataFile'] = testFile

# Enabling Debug Parameters
if debug_mode:
    trueStateTestShape = trueStateTEST.shape
    instantaneousSquaredErrors = torch.empty((trueStateTestShape[0], 
                                              trueStateTestShape[1], 
                                              trueStateTestShape[3]),
                                              dtype=torch.float)
    tcnPredVals = torch.empty((2, trueStateTestShape[0],
                               trueStateTestShape[1],
                               trueStateTestShape[3]),
                               dtype=torch.float)

    realValues = torch.empty((2, trueStateTestShape[0],
                               trueStateTestShape[1],
                               trueStateTestShape[3]),
                               dtype=torch.float)
    measStateTestShape = measuredStateTEST.shape
    obsValues = torch.empty((2, measStateTestShape[0],
                                 measStateTestShape[1], measStateTestShape[3],
                                 measStateTestShape[4]), dtype=torch.float)

trueStateTEST = torch.from_numpy(trueStateTEST)
measuredStateTEST = torch.from_numpy(measuredStateTEST)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ INITIALIZING THE MODEL ~~~~~~~~~~~~~~~~ ###
if not testSession:
    # Saving the model parameters so we can use them to load the model properly later
    modelParameters = {
        'input_channels': input_channels,
        'n_classes': n_classes,
        'channel_sizes': channel_sizes,
        'kernel_size': kernel_size,
        'dropout': dropout
    }
else:
    input_channels = modelContext['model_parameters']['input_channels']
    n_classes = modelContext['model_parameters']['n_classes']
    channel_sizes = modelContext['model_parameters']['channel_sizes']
    kernel_size = modelContext['model_parameters']['kernel_size']
    dropout = modelContext['model_parameters']['dropout']
# Generate the model
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

# Creating a backup of the model that we can use for early stopping
modelBEST = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

# Setting up the biases variable
if (not testSession) and biasRemoval:
    biases_TandE = torch.zeros(measuredStateTRAIN[:,:,0,0].shape)


# Logic Structure of using cuda in either multiple GPU or single GPU 
# orientations
if(args.cuda):
    if(torch.cuda.is_available()):
        # Use every GPU available
        if(args.cuda_device == 'all'):
            # If multiple GPU's are available
            if(torch.cuda.device_count() > 1):
                print('using multiple GPU\'s')
                model = nn.DataParallel(model)
                modelBEST = nn.DataParallel(modelBEST)
                for m in range(0, torch.cuda.device_count()):
                    device = torch.device('cuda:' + str(m))
                    model.to(device)
                    modelBEST.to(device)
            # If only a single GPU is available then do nothing because this is 
            # default behaviour for cuda

        # If a device ID was specified, then use that instead
        else:
            try:
                cuda_device = int(args.cuda_device)
            except:
                raise ValueError('Need to specify an integer or all for ' +
                                 '--cuda_device argument')
            device = torch.device('cuda:' + str(cuda_device))
            model.to(device)
            modelBEST.to(device)
                
            
    else:
        raise ValueError('cuda not available, --cuda unavailable')

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ LOAD DATA INTO CUDA ~~~~~~~~~~~~~~~~~~~ ###

if args.cuda:

    model.cuda()
    modelBEST.cuda()
    # If we are not just testing then load everything into cuda
    if not testSession:

        # Train set
        trueStateTRAIN = trueStateTRAIN.cuda()
        measuredStateTRAIN = measuredStateTRAIN.cuda()

        # Evaluation set
        trueStateEVAL = trueStateEVAL.cuda()
        measuredStateEVAL = measuredStateEVAL.cuda()

        if biasRemoval:
            biases_TandE = biases_TandE.cuda()

    # Pushing the Squared Error calculations to cuda as well
    if debug_mode:
        instantaneousSquaredErrors = instantaneousSquaredErrors.cuda()
    # Test set
    trueStateTEST = trueStateTEST.cuda()
    measuredStateTEST = measuredStateTEST.cuda()

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ OPTIMIZER ~~~~~~~~~~~~~~~~~~~~~~~~ ###
if not testSession:
    # Create the optimizer
    optimizerParameters = {
        'optim': optimMethod,
        'lr': lr
    }
else:
    # Loading the optimizer parameters to use
    optimMethod = modelContext['optimizer_parameters']['optim']
    lr = modelContext['optimizer_parameters']['lr']

# Initializing the optimizer
optimizer = getattr(optim, optimMethod)(model.parameters(), lr=lr)

# Creating a learning rate scheduler that updates the learning rate when the model plateaus
# Divides the learning rate by 2 if the model has not gotten a lower total loss in 10 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=19)

# Defining index that will only grab the predicted values
predInds = [1,3]
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def train(epoch):

    # Initialize training model and parameters
    model.train()
    total_loss = 0
    trainLoss = 0

    # Shuffle the data each epoch
    #[shuffledMeasTRAIN, perm] = shuffleMeasTrainData(measuredStateTRAIN)
    #shuffledTrueTRAIN = shuffleTrueTrainData(trueStateTRAIN, perm)
 ################################

    # Training loop - run until we process every series of data
    for i in range(0, trainSeriesLength):

        # Grab the current series
        #x = shuffledMeasTRAIN[:, :, :, i]
        #y = shuffledTrueTRAIN[:, predInds, i]
        x = measuredStateTRAIN[:,:,:,i]
        y = trueStateTRAIN[:, predInds, i]
        
        if(args.cuda):
            x = x.cuda()
            y = y.cuda()
        # Subtracting the bias from each of the samples
        if biasRemoval:
            x = torch.flip(x, [2])
            biases_TandE[:,:] = x[:, :, 0]
            x = x - biases_TandE[:, :, None]

        x = x.float()
        y = y.float()

        # Forward/backpass
        optimizer.zero_grad()
        # Computing the output of the network and adding the bias back in
        if biasRemoval:
            output = model(x) + biases_TandE
        else:
            output = model(x)
        loss = F.mse_loss(output, y, reduction="sum")
        loss.backward()

        # Gradient clipping option
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        total_loss += loss.item()
        trainLoss += loss.item()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display training results
        if i % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = i

            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, measuredStateTRAIN.size(3), 100. * processed / measuredStateTRAIN.size(3), lr, cur_loss))
            PredMSE = torch.sum((output[:, 0] - y[:, 0]) ** 2 + (output[:, 1] - y[:, 1]) ** 2) / output.size(0)
            # TrueMSE = torch.sum((output[:, 0] - y[:, 0]) ** 2 + (output[:, 2] - y[:, 2]) ** 2) / output.size(0)
            total_loss = 0
    print('total loss over the whole training set was {}'.format(trainLoss/trainDataLen))
    finalLoss = trainLoss/trainDataLen
    return finalLoss
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~ EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~ ###

def evaluate():

    # MSE over the entire series of data
    TotalAvgPredMSE = 0
    # TotalAvgTrueMSE = 0
    n = 0
    totalLoss=0

    # Pre-allocating space for the tensor that will contain the MSEs of each batch from our test set
    evalPredMSEs = torch.empty(measuredStateEVAL.size(3))
    # evalEstMSEs = torch.empty(measuredStateEVAL.size(3))

    # Training loop - run until we only have one batch size of data left
    for i in range(0, (measuredStateEVAL.size(3))):

        # Grab the current series
        x_eval = measuredStateEVAL[:, :, :, i]
        y_eval = trueStateEVAL[:, predInds, i]

        if args.cuda:
            x_eval = x_eval.cuda()
            y_eval = y_eval.cuda()
        # Subtracting the bias from each of the samples
        if biasRemoval:
            x_eval = torch.flip(x_eval, [2])
            biases_TandE[:,:] = x_eval[:, :, 0]
            x_eval = x_eval - biases_TandE[:, :, None].type(torch.float64)

        x_eval = x_eval.float()
        y_eval = y_eval.type(torch.double)

        # Model eval setting
        model.eval()
        with torch.no_grad():

            # Compute output and loss
            if biasRemoval:
                output = model(x_eval).type(torch.float64) + biases_TandE.type(torch.float64)
            else:
                output = model(x_eval).type(torch.float64)
            
            eval_loss = F.mse_loss(output, y_eval, reduction="sum")

            PredMSE = torch.sum((output[:, 0] - y_eval[:, 0]) ** 2 + (output[:, 1] - y_eval[:, 1]) ** 2) / output.size(0)
            # TrueMSE = torch.sum((output[:, 0] - y_eval[:, 0]) ** 2 + (output[:, 2] - y_eval[:, 2]) ** 2) / output.size(0)
            # evalEstMSEs[i] = TrueMSE
            evalPredMSEs[i] = PredMSE
            TotalAvgPredMSE+=PredMSE
            # TotalAvgTrueMSE+=TrueMSE
            totalLoss +=  eval_loss.item()
        n+=1

    totalLoss = totalLoss / (n * batch_size)
    TotalAvgPredMSE = TotalAvgPredMSE / n
    # TotalAvgTrueMSE = TotalAvgTrueMSE / n

    #predVariance = torch.sum((evalPredMSEs[:]-TotalAvgPredMSE)**2)/evalPredMSEs.size(0)
    # estVariance = torch.sum((evalEstMSEs[:]-TotalAvgTrueMSE)**2)/evalEstMSEs.size(0)

    print('TotalAvgPredMSE: ', TotalAvgPredMSE.item())
    # print('TotalAvgTrueMSE: ', TotalAvgTrueMSE.item())

    return totalLoss


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~ TEST ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def test():
    # Looping through each of the sets of data with different AR Coefficients
    for r in range(0, testSetLen):
        # Total MSE
        TotalAvgPredMSE = 0
        # TotalAvgTrueMSE = 0
        n = 0
        # Pre-allocating space for the tensor that will contain the MSEs of
        # each batch from our training set
        testPredMSEs = torch.empty(testSeriesLength)
        # testEstMSEs = torch.empty(testSeriesLength)
        # Training loop - run until we only have one batch size of data left
        for i in range(0, testSeriesLength):

            # Grab the current series
            x_test = measuredStateTEST[r, :, :, :, i]
            y_test = trueStateTEST[r, :, predInds, i]


            if args.cuda:
                x_test = x_test.cuda()
                y_test = y_test.cuda()
            
            # Subtracting the bias from each of the samples
            if biasRemoval:
                x_test = torch.flip(x_test, [2])
                biases = x_test[:, :, 0]
                x_test = x_test - biases[:, :, None]
            

            x_test = x_test.float()
            y_test = y_test.float()

            # Model test
            modelBEST.eval()
            with torch.no_grad():

                # Compute output and loss
                if biasRemoval:
                    output = modelBEST(x_test) + biases.type(torch.float32)
                    
                else:
                    output = modelBEST(x_test)

                test_loss = F.mse_loss(output, y_test, reduction="sum")
                PredMSE = torch.sum(((output[:, 0] - y_test[:, 0]) ** 2) + (output[:, 1] - y_test[:, 1]) ** 2) / output.size(0)

                if debug_mode:
                    instantaneousSquaredErrors[r, :, i] = ((output[:, 0] - y_test[:, 0]) ** 2) + \
                                                          ((output[:, 1] - y_test[:, 1]) ** 2)
                    # Grabbing complex and real predicted numbers
                    tcnPredVals[0, r, :, i] = output[:, 0]
                    tcnPredVals[1, r, :, i] = output[:, 1]
                    
                    realValues[0, r, :, i] = y_test[:, 0]
                    realValues[1, r, :, i] = y_test[:, 1]
                    # Setting the observed states back to what they
                    # were before preprocessing
                    if biasRemoval:
                        x_test = x_test.type(torch.float64) + biases[:, :, None].type(torch.float64)

                    obsValues[0, r, :, :, i] = x_test[:, 0, :]
                    obsValues[1, r, :, :, i] = x_test[:, 1, :]


                # TrueMSE = torch.sum((output[:, 0] - y_test[:, 0]) ** 2 + (output[:, 2] - y_test[:, 2]) ** 2) / output.size(0)
                TotalAvgPredMSE+=PredMSE
                # TotalAvgTrueMSE+=TrueMSE
                testPredMSEs[i] = PredMSE
                # testEstMSEs[i] = TrueMSE

            n+=1
        # Recording the instantaneous errors if the model is set to debug mode
        if debug_mode:
            testDataInfo[r][u'instantaneousSquaredErrors'] = (torch.squeeze(instantaneousSquaredErrors[r,:,:])).cpu().numpy()
            testDataInfo[r][u'tcnPredVals'] = (torch.squeeze(tcnPredVals[:,r,:,:])).cpu().numpy()

            testDataInfo[r][u'realValues'] = (torch.squeeze(realValues[:,r,:,:])).cpu().numpy()
            testDataInfo[r][u'obsValues'] = (torch.squeeze(obsValues[:,r,:,:,:])).cpu().numpy()
        TotalAvgPredMSE = TotalAvgPredMSE / n
        # TotalAvgTrueMSE = TotalAvgTrueMSE / n

        # predVariance = torch.sum((testPredMSEs[:]-TotalAvgPredMSE)**2)/testPredMSEs.size(0)
        # estVariance = torch.sum((testEstMSEs[:]-TotalAvgTrueMSE)**2)/testEstMSEs.size(0)

        # Logging
        # testDataInfo[r][u'predictionVariance'] = predVariance.item()
        # testDataInfo[r][u'estimationVariance'] = estVariance.item()
        # testDataInfo[r][u'estimationMSE'] = TotalAvgTrueMSE.item()
        testDataInfo[r][u'predictionMSE'] = TotalAvgPredMSE.item()

        print('TCN Performance:')
        print("TotalAvgPredMSE for test set number {}: ".format(r+1), TotalAvgPredMSE.item())
        # print("TotalAvgTrueMSEfor test set number {}: ".format(r+1), TotalAvgTrueMSE.item())
        # Commenting out printing the variances, since they serve no purpose as of now
        # print("Variance of Prediction for test set number {}: ".format(r+1), predVariance.item())
        # print("Variance of Estimate for test set number {}: ".format(r+1), estVariance.item())

        # Computing the LS performance on this data set
        LS_MSEE, LS_MSEP = LSTesting(LSCoefficients, LSandKFTestData[r])
        LS_Results = LSTesting(LSCoefficients, LSandKFTestData[r], debug_mode)
        
        LS_MSEE = LS_Results[0]
        LS_MSEP = LS_Results[1]
        
        if debug_mode:
            testDataInfo[r][u'LSInstaErrs'] = LS_Results[3]
            testDataInfo[r][u'LSPredVals'] = LS_Results[2]
        
        print('LS Performance')
        print("MSE of LS predictor for set number {}: ".format(r+1), LS_MSEP)
        print("MSE of LS estimator for set number {}: ".format(r+1), LS_MSEE)
        
        testDataInfo[r][u'LS_PredMSE'] = LS_MSEP
        testDataInfo[r][u'LS_EstMSE'] = LS_MSEE
        
        # Computing Kalman performance
        KFResults = KFTesting(LSandKFTestData[r], KFARCoeffs, initTest=args.initTest, debug=debug_mode)
        
        if debug_mode:
            testDataInfo[r][u'KFInstaErrs'] = KFResults[2]
            testDataInfo[r][u'KFPredVals'] = KFResults[3]

        KF_MSEP = KFResults[1]
        KF_MSEE = KFResults[0]
        

        print('KF Performance')
        print("MSE of KF predictor for set number {}: ".format(r+1), KF_MSEP)
        print("MSE of KF estimator for set number {}: ".format(r+1), KF_MSEE)

        testDataInfo[r][u'KF_PredMSE'] = KF_MSEP
        testDataInfo[r][u'KF_EstMSE'] = KF_MSEE

        print('Riccati Convergence MSE')
        print("MSE Riccati Prediction for set number {}: ".format(r+1), testDataInfo[r]['riccatiConvergencePred'])

        # Printing a newline to make it easier to tell test sets apart
        print(' ')


    return test_loss.item()
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~~ LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

if not testSession:
    # Determine where to save model checkpoints and final model parameters
    fileSpaceFound = False
    modelNumber = 0

    # Create the directory to save models to, if non-existent
    if not (path.exists('./models')):
        os.mkdir('./models', 0o755)

    # Saving model to a new file
    while (fileSpaceFound == False):
        modelNumber += 1
        modelPath = './models/model' + str(modelNumber) + '.pth'
        if not (path.exists(modelPath)):
            print('model parameters will be saved to: ', modelPath)
            fileSpaceFound = True

    # Initializing model context dict to be saved with model
    modelContext = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_parameters': modelParameters,
        'optimizer_parameters': optimizerParameters,
        'LSCoefficients': {}
    }

    # Training the Least Squares so we can evaluate its performance on the same dataset
    LSCoefficients = LSTraining(LSTrainData)
    # Saving the LS Coefficients so we do not need to train it again
    modelContext['LSCoefficients'] = LSCoefficients

    # How many times in a row we have had a worse loss than our best case loss scenario
    numEpochsSinceBest = 0
    # Letting the model know when the last epoch happens so we can record the MSEs of the individual samples
    for ep in range(0, epochs):
        train(ep)
        tloss = evaluate()
        
        scheduler.step(tloss)
        # Updating the learning rate that will be displayed for each epoch
        lr = optimizer.param_groups[0]['lr']

        # Run through all epochs, find the best model and save it for testing
        if(ep == 0):
            bestloss = tloss
            modelBEST.load_state_dict(model.state_dict())
            modelContext['model_state_dict'] = model.state_dict()
            modelContext['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(modelContext, modelPath)
            print('model saved at {}'.format(modelPath))
        else:
            if(tloss <= bestloss):
                numEpochsSinceBest = 0
                bestloss = tloss
                modelBEST.load_state_dict(model.state_dict())
                modelContext['model_state_dict'] = model.state_dict()
                modelContext['optimizer_state_dict'] = optimizer.state_dict()
                modelContext['epoch'] = ep
                torch.save(modelContext, modelPath)
                print("better loss")
                print('model saved at {}'.format(modelPath))
            else:
                numEpochsSinceBest += 1
                print("worse loss: {} epochs since best loss".format(numEpochsSinceBest))
                if(numEpochsSinceBest >= 43):
                    print('No progress made in 43 epochs, model is over fitting')
                    break
               # What this does is reset the model back to the best model after 10 epochs of no improvement
               # to get the benefit of the decreased step size
                if((numEpochsSinceBest % 20) == 0):
                    model.load_state_dict(modelBEST.state_dict())
                    print('model reset to best model')

    print('model saved')
    torch.save(modelContext, modelPath)
else:
    # If we are just loading and testing a model, then we set the path properly to be saved
    modelPath = args.model_path
    modelBEST.load_state_dict(modelContext['model_state_dict'])
    optimizer.load_state_dict(modelContext['optimizer_state_dict'])

    # Loading the LS Coefficients
    LSCoefficients = modelContext['LSCoefficients']


# Test once we are done training the model
tloss = test()

print("check check")

# End timing
end = time.time()
simRunTime=(end-start)
print('this simulation took:', simRunTime, 'seconds to run')

fileContent[u'testInfo'] = testDataInfo
fileContent[u'modelPath'] = modelPath
fileContent[u'trainingLength(seconds)'] = simRunTime
fileContent[u'runType'] = 'TCN'

if(altAlgs):
    fileContent[u'IMMPredVals'] = IMMPredVals
    fileContent[u'GKFPredVals'] = GKFPredVals

print('log data saved to: ', logName)
print('model parameters saved to: ', modelPath)

hdf5s.savemat(logName, fileContent)