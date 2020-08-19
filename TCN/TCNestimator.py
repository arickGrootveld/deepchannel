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
import os
from os import path
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model import TCN
from mismatch_data_gen import ARDatagenMismatch
from utilities import convertToBatched, matSave, shuffleMeasTrainData, shuffleTrueTrainData
from LeastSquares.LSFunction import LSTesting, LSTraining
from KalmanFilter.KFFunction import KFTesting2 

# Timer for logging how long the training takes to execute
import time
start = time.time()

# Importing file handler for .mat files
import hdf5storage as hdf5s

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
parser.add_argument('--dropout', type=float, default=0.15,
                    help='dropout applied to layers (default: 0.15)')

# Gradient clipping
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')

# Training epochs
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit (default: 1)')

# Kernel (filter) size
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')

# Number of levels in the model
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 5)')

# Sequence length
parser.add_argument('--seq_len', type=int, default=20,
                    help='sequence length (default: 20)')

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
parser.add_argument('--nhid', type=int, default=5,
                    help='number of hidden units per layer (default: 5)')

# Random seed value
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

# Data Generation Length
parser.add_argument('--simu_len', type=float, default=1e2,
                    help='amount of data generated for training (default: 1e2)')

# Amount of sequences used for each set of AR Coefficients tested against
parser.add_argument('--test_set_depth', type=float, default=1e2,
                    help='number of sequences generated per pair of AR Coefficients used for testing (default: 1e2)')
# Number of sets of AR Coefficients generated for testing
parser.add_argument('--test_set_len', type=float, default=10,
                    help='Number of different AR Coefficients that will be used for testing (default=10)')

# Length of data used for eval of models effectiveness
parser.add_argument('--eval_len', type=float, default=1e2,
                    help='amount of data generated for testing (default: 1e2)')

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

# Variance of the random variables that compuse the AR Coefficients
parser.add_argument('--AR_var', type=float, default=0,
                    help='variance of the AR parameters in the data generation (default=0)')

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
seq_length = args.seq_len
epochs = args.epochs
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
lr = args.lr
trainDataLen = int(args.simu_len)
seed = args.seed
optimMethod = args.optim

debug_mode = args.debug

evalDataLen = int(args.eval_len)
testDataLen = int(args.test_set_depth)
AR_var = args.AR_var

trainFile = args.trainDataFile
evalFile = args.evalDataFile
testFile = args.testDataFile

KFARCoeffs = []
for KFCoeff in args.KFCoeffs:
    KFARCoeffs.append(float(KFCoeff))

# Calculating the number of batches that will need to be created given the simulation length and the batch size
trainSeriesLength = int(trainDataLen / batch_size)
evaluationSeriesLength = int(evalDataLen / batch_size)
# Doing the same calculation as above for the test data set
testSeriesLength = int(testDataLen/batch_size)
# The appropriate seriesLength variables get overwritten if we load data from files

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

    if(trainFile == 'None'):
        # Generate AR process training data set - both measured and real states
        trainStateData, trainStateInfo = ARDatagenMismatch([trainDataLen, AR_n, AR_var, seq_length], seed, args.cuda)
        # Convert the data from normal formatting to batches
        trueStateTRAIN, measuredStateTRAIN = convertToBatched(trainStateData[2], trainStateData[1], batch_size)
        fileContent[u'trainDataFile'] = trainStateInfo['filename']
        fileContent[u'trainDataSize'] = trueStateTRAIN.shape
        # TODO: Improve the format at some point in the future, but for now we are just grabbing the trainingData to train
        # TODO: the LS

        # Only grabbing a portion of the train samples for the LS training, because otherwise the pseudo-inverse
        # will use too much data
        if(trainDataLen >= pseudoInverseDataThreshold):
            LSTrainPlaceHolderTrueStates = trainStateData[0][:, :, 0:pseudoInverseDataThreshold]
            LSTrainPlaceHolderMeasuredStates = trainStateData[1][:,:,0:pseudoInverseDataThreshold]
            LSTrainPlaceHolderFinalStates = trainStateData[2][:,0:pseudoInverseDataThreshold]
            # Recreating the data, but only the first 1e6 of it
            LSTrainData = [LSTrainPlaceHolderTrueStates, LSTrainPlaceHolderMeasuredStates,
                           LSTrainPlaceHolderFinalStates]
        else:
            LSTrainData = trainStateData
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
    if(evalFile == 'None'):
        # Generate AR process evaluation data set - both measured and real states
        evalStateData, evalStateInfo = ARDatagenMismatch([evalDataLen, AR_n, AR_var, seq_length], seed + 1, args.cuda)
        # Convert the data from normal formatting to batches
        trueStateEVAL, measuredStateEVAL = convertToBatched(evalStateData[2], evalStateData[1], batch_size)
        fileContent[u'evalDataFile'] = evalStateInfo['filename']
        fileContent[u'evalDataSize'] = trueStateEVAL.shape
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
# Have to do things slightly different here compared to Eval and Train, because we want to
# test against a bunch of different sequences of the same AR Process (same coefficients)
# so we can't just generate a single set of data. We simply generate several sequences
# from the same AR process, and then format everything so it makes sense

testSetLen = int(args.test_set_len)
# trueStateDataTEST Dimensionality: comprised of sets of data based on the number of AR
#                                   coefficients that are to be generated in the 1st
#                                   dimension, then of batch elements in the 2nd
#                                   dimension, real and imaginary portions of the true state
#                                   in the 3rd dimension, and by batches in the 4th dimension
trueStateTEST = np.empty((testSetLen, batch_size, 4, testSeriesLength), dtype=float)

# measuredStateTEST Dimensionality: comprised of sets of data based on the number of AR
#                                   coefficients used for testing in the 1st dimension,
#                                   batch elements in the 2nd dimension, real and complex
#                                   values in the 3rd dimension, sequence elements in the
#                                   4th dimension, and by batches in the 5th dimension

measuredStateTEST = np.empty((testSetLen, batch_size, 2, seq_length, testSeriesLength), dtype=float)




# Loop for generating the data set for each pair AR Coefficients to test against
if(testFile == 'None'):
    LSandKFTestData = []
    testDataInfo = []
    for k in range(0,testSetLen):

        # Generating the data with no variance between sequences (which is what the False in the array is for)
        # because we want to generate data with a single set of AR Coefficients
        subsetTestStateData, subsetTestDataInfo = ARDatagenMismatch([testDataLen, AR_n, AR_var, seq_length, False], seed + 2 + k, args.cuda)
        trueStateTEST[k,:,:,:], measuredStateTEST[k,:,:,:,:] = convertToBatched(subsetTestStateData[2], subsetTestStateData[1], batch_size)

        # Storing the data that the Least Squares and Kalman Filter will be using
        LSandKFTestData.append(subsetTestStateData)

        subsetInfoHolder = {}
        # Grabbing the first riccatiConvergence because they should all be the same for both the estimate and prediction
        subsetInfoHolder[u'riccatiConvergencePred'] = subsetTestDataInfo['riccatiConvergences'][0,0]
        subsetInfoHolder[u'riccatiConvergenceEst'] = subsetTestDataInfo['riccatiConvergences'][1,0]
        # Grabbing the first set of AR Coefficients from the F matrix because they should all be the same
        subsetInfoHolder[u'ARCoefficients'] = subsetTestDataInfo['trueFs'][0,:,0]
        # Grabbing the file path of the data file
        subsetInfoHolder[u'dataFilePath'] = subsetTestDataInfo['filename']
        subsetInfoHolder[u'seed'] = subsetTestDataInfo['seed']
        testDataInfo.append(subsetInfoHolder)
    # Saving relevant data so it can be recovered and reused
    testDataToBeSaved = {}
    testDataToBeSaved[u'trueStateTEST'] = trueStateTEST
    testDataToBeSaved[u'measuredStateTEST'] = measuredStateTEST
    testDataToBeSaved[u'testDataInfo'] = testDataInfo
    testDataToBeSaved[u'LSandKFTestData'] = LSandKFTestData
    testFile = matSave('data', 'testData', testDataToBeSaved)
    fileContent[u'testDataFile'] = testFile
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

# Setting up the biases variables
if not testSession:
    biases_TandE = torch.zeros(measuredStateTRAIN[:,:,0,0].shape)

biases_Test = torch.zeros(measuredStateTEST[0,:,:,0,0].shape)

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
                for m in range(0, torch.cuda.device_count()):
                    device = torch.device('cuda:' + str(m))
                    model.to(device)
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
                
            
    else:
        raise ValueError('cuda not available, --cuda unavailable')

# Creating a backup of the model that we can use for early stopping
modelBEST = model
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ LOAD DATA INTO CUDA ~~~~~~~~~~~~~~~~~~~ ###

if args.cuda:

    model.cuda()
    modelBEST.cuda()
    biases_Test = biases_Test.cuda()
    # If we are not just testing then load everything into cuda
    if not testSession:

        # Train set
        trueStateTRAIN = trueStateTRAIN.cuda()
        measuredStateTRAIN = measuredStateTRAIN.cuda()

        # Evaluation set
        trueStateEVAL = trueStateEVAL.cuda()
        measuredStateEVAL = measuredStateEVAL.cuda()

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=9)

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
    [shuffledMeasTRAIN, perm] = shuffleMeasTrainData(measuredStateTRAIN)
    shuffledTrueTRAIN = shuffleTrueTrainData(trueStateTRAIN, perm)
 ################################

    # Training loop - run until we process every series of data
    for i in range(0, trainSeriesLength):

        # Grab the current series
        x = shuffledMeasTRAIN[:, :, :, i]
        y = shuffledTrueTRAIN[:, predInds, i]
        if(args.cuda):
            x = x.cuda()
            y = y.cuda()
        # Subtracting the bias from each of the samples
        biases_TandE[:,:] = x[:, :, 0]

        x = x - biases_TandE[:, :, None]

        x = x.float()
        y = y.float()

        # Forward/backpass
        optimizer.zero_grad()
        # Computing the output of the network and adding the bias back in
        output = model(x) + biases_TandE
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
            print('PredMSE: ', PredMSE)
            # print('TrueMSE: ', TrueMSE)
            total_loss = 0
    print('total loss over the whole training set was {}'.format(trainLoss/trainDataLen))

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

        # Subtracting the bias from each of the samples
        biases_TandE[:,:] = x_eval[:, :, 0]
        x_eval = x_eval - biases_TandE[:, :, None].type(torch.float64)

        x_eval = x_eval.float()
        y_eval = y_eval.type(torch.double)

        # Model eval setting
        model.eval()
        with torch.no_grad():

            # Compute output and loss
            output = model(x_eval).type(torch.float64) + biases_TandE.type(torch.float64)
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

            # Subtracting the bias from each of the samples
            biases = x_test[:, :, 0]
            x_test = x_test - biases[:, :, None]

            x_test = x_test.float()
            y_test = y_test.float()

            # Model test
            modelBEST.eval()
            with torch.no_grad():

                # Compute output and loss
                output = modelBEST(x_test) + biases.type(torch.float32)
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
        KFResults = KFTesting2(LSandKFTestData[r], KFARCoeffs, initTest=args.initTest, debug=debug_mode)
        
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
            modelContext['model_state_dict'] = model.state_dict()
            modelContext['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(modelContext, modelPath)
            print('model saved at {}'.format(modelPath))
        else:
            if(tloss <= bestloss):
                numEpochsSinceBest = 0
                bestloss = tloss
                modelBEST = model
                modelContext['model_state_dict'] = model.state_dict()
                modelContext['optimizer_state_dict'] = optimizer.state_dict()
                modelContext['epoch'] = ep
                torch.save(modelContext, modelPath)
                print("better loss")
                print('model saved at {}'.format(modelPath))
            else:
                numEpochsSinceBest += 1
                print("worse loss: {} epochs since best loss".format(numEpochsSinceBest))
                if(numEpochsSinceBest >= 31):
                    print('No progress made in 31 epochs, model is over fitting')
                    break
                # What this does is reset the model back to the best model after 10 epochs of no improvement
                # to get the benefit of the decreased step size
                if((numEpochsSinceBest % 10) == 0):
                    model = modelBEST
                    print('model reset to best model')



        print(tloss)

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

print('log data saved to: ', logName)
print('model parameters saved to: ', modelPath)

hdf5s.savemat(logName, fileContent)
