# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#            TCN Based Channel Estimation Neural Network Training
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import torch
import argparse
from argparse import RawTextHelpFormatter
import torch.optim as optim
import torch.nn.functional as F
import os
from os import path
import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model import TCN
from mismatch_data_gen import ARDatagenMismatch
from utilities import convertToBatched, matSave
from LeastSquares.LSFunction import LSTesting, LSTraining
from KalmanFilter.KFFunction import KFTesting

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

# CUDA device
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device to use if running on cuda, only works if cuda'
                    'is enabled (default 0)')

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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval (default: 200')

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
n_classes = 4

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

evalDataLen = int(args.eval_len)
testDataLen = int(args.test_set_depth)
cuda_device = args.cuda_device
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
    modelContext = torch.load(args.model_path)
modelPath = args.model_path
### ~~~~~~~~~~~~~~~ LOAD DATA/GENERATE MODEL ~~~~~~~~~~~~~~~~ ###
# Here we have the option of loading from a saved .mat file or just calling the data generation
# function explicitly

# AR data generation parameters
AR_n = 2

# ~~~~~~~~~~~~~~~~~~ LOAD TRAINING SET
if not testSession:
    
    
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
    trainDataLen = trainSeriesLength * trueStateTRAIN.shape[0]

    # Convert numpy arrays to tensors
    trueStateTRAIN = torch.from_numpy(trueStateTRAIN)
    measuredStateTRAIN = torch.from_numpy(measuredStateTRAIN)

    # ~~~~~~~~~~~~~~~~~~ LOAD EVALUATION SET
    
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




# Loading all test data from a file
testDataToBeLoaded = hdf5s.loadmat(testFile)
trueStateTEST = testDataToBeLoaded['trueStateTEST']
measuredStateTEST = testDataToBeLoaded['measuredStateTEST']

# overriding parameters from command line, because we are loading from file
testSetLen = trueStateTEST.shape[0]
testSeriesLength = trueStateTEST.shape[3]

print('test data loaded from: {}'.format(testFile))
fileContent[u'testDataFile'] = testFile

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
modelBEST = model
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ LOAD DATA INTO CUDA ~~~~~~~~~~~~~~~~~~~ ###

if args.cuda:

    torch.cuda.set_device(cuda_device)
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

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def train(epoch):

    # Initialize training model and parameters
    model.train()
    total_loss = 0

 ################################

    # Training loop - run until we process every series of data
    for i in range(0, trainSeriesLength):

        # Grab the current series
        x = measuredStateTRAIN[:, :, :, i]
        y = trueStateTRAIN[:, :, i]

        x = x.float()
        y = y.float()

        # Forward/backpass
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y, reduction="sum")
        loss.backward()

        # Gradient clipping option
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        total_loss += loss.item()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display training results
        if i % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = i

            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, measuredStateTRAIN.size(3), 100. * processed / measuredStateTRAIN.size(3), lr, cur_loss))
            PredMSE = torch.sum((output[:, 1] - y[:, 1]) ** 2 + (output[:, 3] - y[:, 3]) ** 2) / output.size(0)
            TrueMSE = torch.sum((output[:, 0] - y[:, 0]) ** 2 + (output[:, 2] - y[:, 2]) ** 2) / output.size(0)
            print('PredMSE: ', PredMSE)
            print('TrueMSE: ', TrueMSE)
            total_loss = 0

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~ EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~ ###

def evaluate():

    # MSE over the entire series of data
    TotalAvgPredMSE = 0
    TotalAvgTrueMSE = 0
    n = 0
    totalLoss=0

    # Pre-allocating space for the tensor that will contain the MSEs of each batch from our test set
    evalPredMSEs = torch.empty(measuredStateEVAL.size(3))
    evalEstMSEs = torch.empty(measuredStateEVAL.size(3))

    # Training loop - run until we only have one batch size of data left
    for i in range(0, (measuredStateEVAL.size(3))):

        # Grab the current series
        x_eval = measuredStateEVAL[:, :, :, i]
        y_eval = trueStateEVAL[:, :, i]

        x_eval = x_eval.float()
        y_eval = y_eval.float()

        # Model eval setting
        model.eval()
        with torch.no_grad():

            # Compute output and loss
            output = model(x_eval)
            eval_loss = F.mse_loss(output, y_eval, reduction="sum")

            PredMSE = torch.sum((output[:, 1] - y_eval[:, 1]) ** 2 + (output[:, 3] - y_eval[:, 3]) ** 2) / output.size(0)
            TrueMSE = torch.sum((output[:, 0] - y_eval[:, 0]) ** 2 + (output[:, 2] - y_eval[:, 2]) ** 2) / output.size(0)
            evalEstMSEs[i] = TrueMSE
            evalPredMSEs[i] = PredMSE
            TotalAvgPredMSE+=PredMSE
            TotalAvgTrueMSE+=TrueMSE
            totalLoss +=  eval_loss.item()
        n+=1

    totalLoss = totalLoss / (n * batch_size)
    TotalAvgPredMSE = TotalAvgPredMSE / n
    TotalAvgTrueMSE = TotalAvgTrueMSE / n

    predVariance = torch.sum((evalPredMSEs[:]-TotalAvgPredMSE)**2)/evalPredMSEs.size(0)
    estVariance = torch.sum((evalEstMSEs[:]-TotalAvgTrueMSE)**2)/evalEstMSEs.size(0)

    print('TotalAvgPredMSE: ', TotalAvgPredMSE.item())
    print('TotalAvgTrueMSE: ', TotalAvgTrueMSE.item())

    return totalLoss


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~ TEST ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def test():
    # Looping through each of the sets of data with different AR Coefficients
    for r in range(0, testSetLen):
        # Total MSE
        TotalAvgPredMSE = 0
        TotalAvgTrueMSE = 0
        n = 0
        # Pre-allocating space for the tensor that will contain the MSEs of
        # each batch from our training set
        testPredMSEs = torch.empty(testSeriesLength)
        testEstMSEs = torch.empty(testSeriesLength)
        # Training loop - run until we only have one batch size of data left
        for i in range(0, testSeriesLength):

            # Grab the current series
            x_test = measuredStateTEST[r, :, :, :, i]
            y_test = trueStateTEST[r, :, :, i]

            x_test = x_test.float()
            y_test = y_test.float()

            # Model test
            modelBEST.eval()
            with torch.no_grad():

                # Compute output and loss
                output = modelBEST(x_test)
                test_loss = F.mse_loss(output, y_test, reduction="sum")

                PredMSE = torch.sum((output[:, 1] - y_test[:, 1]) ** 2 + (output[:, 3] - y_test[:, 3]) ** 2) / output.size(0)
                TrueMSE = torch.sum((output[:, 0] - y_test[:, 0]) ** 2 + (output[:, 2] - y_test[:, 2]) ** 2) / output.size(0)
                TotalAvgPredMSE+=PredMSE
                TotalAvgTrueMSE+=TrueMSE
                testPredMSEs[i] = PredMSE
                testEstMSEs[i] = TrueMSE

            n+=1

        TotalAvgPredMSE = TotalAvgPredMSE / n
        TotalAvgTrueMSE = TotalAvgTrueMSE / n

        predVariance = torch.sum((testPredMSEs[:]-TotalAvgPredMSE)**2)/testPredMSEs.size(0)
        estVariance = torch.sum((testEstMSEs[:]-TotalAvgTrueMSE)**2)/testEstMSEs.size(0)

        # Logging
        testDataInfo[r][u'predictionVariance'] = predVariance.item()
        testDataInfo[r][u'estimationVariance'] = estVariance.item()
        testDataInfo[r][u'estimationMSE'] = TotalAvgTrueMSE.item()
        testDataInfo[r][u'predictionMSE'] = TotalAvgPredMSE.item()

        print('TCN Performance:')
        print("TotalAvgPredMSE for test set number {}: ".format(r+1), TotalAvgPredMSE.item())
        print("TotalAvgTrueMSEfor test set number {}: ".format(r+1), TotalAvgTrueMSE.item())
        # Commenting out printing the variances, since they serve no purpose as of now
        # print("Variance of Prediction for test set number {}: ".format(r+1), predVariance.item())
        # print("Variance of Estimate for test set number {}: ".format(r+1), estVariance.item())

        # Computing the LS performance on this data set
        LS_MSEE, LS_MSEP = LSTesting(LSCoefficients, LSandKFTestData[r])

        print('LS Performance')
        print("MSE of LS predictor for set number {}: ".format(r+1), LS_MSEP)
        print("MSE of LS estimator for set number {}: ".format(r+1), LS_MSEE)

        testDataInfo[r][u'LS_PredMSE'] = LS_MSEP
        testDataInfo[r][u'LS_EstMSE'] = LS_MSEE

        # Computing Kalman performance
        KF_MSEE, KF_MSEP = KFTesting(LSandKFTestData[r],KFARCoeffs)

        print('KF Performance')
        print("MSE of KF predictor for set number {}: ".format(r+1), KF_MSEP)
        print("MSE of KF estimator for set number {}: ".format(r+1), KF_MSEE)

        testDataInfo[r][u'KF_PredMSE'] = KF_MSEP
        testDataInfo[r][u'KF_EstMSE'] = KF_MSEE

        print('Riccati Convergence MSE')
        print("MSE Riccati Prediction for set number {}: ".format(r+1), testDataInfo[r]['riccatiConvergencePred'])
        print("MSE Riccati Estimation for set number {}: ".format(r+1), testDataInfo[r]['riccatiConvergenceEst'])

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
    # Letting the model know when the last epoch happens so we can record the MSEs of the individual samples
    for ep in range(1, epochs+1):
        train(ep)
        tloss = evaluate()

        # Run through all epochs, find the best model and save it for testing
        if(ep == 1):
            bestloss = tloss
            modelContext['model_state_dict'] = model.state_dict()
            modelContext['optimizer_state_dict'] = optimizer.state_dict()
            modelContext['epoch'] = ep
            torch.save(modelContext, modelPath)
            print('model saved at {}'.format(modelPath))
        else:
            if(tloss <= bestloss):
                bestloss = tloss
                modelBEST = model
                modelContext['model_state_dict'] = model.state_dict()
                modelContext['optimizer_state_dict'] = optimizer.state_dict()
                modelContext['epoch'] = ep
                torch.save(modelContext, modelPath)
                print('model saved at {}'.format(modelPath))
                print("better loss")
            else:
                print("worse loss")
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
