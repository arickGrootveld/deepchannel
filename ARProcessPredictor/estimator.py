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
from utilities import convertToBatched

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

# Data Generation Length
parser.add_argument('--simu_len', type=float, default=1e5,
                    help='amount of data generated for training (default: 1e3)')

# Length of data used for Evaluation of models effectiveness
parser.add_argument('--test_len', type=float, default=1e5,
                    help='amount of data generated for testing (default: 1e3)')

# Load data to train with from data file
parser.add_argument('--trainDataFile', type=str, default='None',
                    help='file path to load training data from, if None then it will generate its own data '
                         '(default=\'None\') \r\n{if a data file is loaded from, the --simLength, --batch_size, '
                         '--seq_len, --AR_var, and --seed parameters \r\nwill do nothing, because these values are used for data '
                         'generation. \r\nThe same applies to the --trainDataFile argument}')

# Load data to test models with from data file
parser.add_argument('--testDataFile', type=str, default='None',
                    help='file path to load test data from, if None then it will generate its own data (default=\'None\')')

parser.add_argument('--AR_var', type=float, default=0.1,
                    help='variance of the AR parameters in the data generation (default=0.1)')


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
logStart='/logs'
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
simu_len = int(args.simu_len)
seed = args.seed
testDataLen = int(args.test_len)
cuda_device = args.cuda_device
AR_var = args.AR_var

trainFile = args.trainDataFile
testFile = args.testDataFile

# Calculating the number of batches that will need to be created given the simulation length and the batch size
trainSeriesLength = int(simu_len/batch_size)

# Doing the same calculation as above for the test data set
testSeriesLength = int(testDataLen/batch_size)
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~ LOAD DATA/GENERATE MODEL ~~~~~~~~~~~~~~~~ ###
# Here we have the option of loading from a saved .mat file or just calling the data generation
# function explicitly

# AR data generation parameters
AR_n = 2


# ~~~~~~~~~~~~~~~~~~ LOAD TRAINING SET
if(trainFile == 'None'):
    # Generate AR process training data set - both measured and real states
    trainStateData, trainStateInfo = ARDatagenMismatch([simu_len, AR_n, AR_var, seq_length], seed, args.cuda)
    # Convert the data from normal formatting to batches
    trueState, measuredState = convertToBatched(trainStateData[2], trainStateData[1], batch_size)

    # for i in range(trainSeriesLength):
    #     # Swapping rows and columns to make the dimensions match
    #     trueState[:,:,i] = np.transpose(trainStateData[2][:,i*batch_size:(i+1)*batch_size])
    #     # Swapping rows and columns, then having to swap the 2nd and 3rd dimensions to make dimensionality match
    #     measuredState[:,:,:,i] = np.swapaxes(np.transpose(trainStateData[1][:,:,i*batch_size:(1+i)*batch_size]), 1, 2)

    #  #  #  #  #  #  #
    # No longer need to log the data to the log file, because the name of the data file that we got the data from will
    # be saved
    #
    # Logging the train data
    # fileContent[u'trainDataActual'] = trueState
    # fileContent[u'trainDataMeas'] = measuredState
    #  #  #  #  #  #

# loading data from file
else:
    # Grab the data from the .mat file
    trainDataDict = hdf5s.loadmat(trainFile)

    # Convert the loaded data into batches for the TCN to run with
    trueState, measuredState = convertToBatched(trainDataDict['finalStateValues'], trainDataDict['observedStateValues'],
                                                batch_size)

    #  #  #  #  #  #
    # measuredState = trainDataDict['measuredData']
    # trueState = trainDataDict['predAndCurState']
    #  #  #  #  #  #

# Convert numpy arrays to tensors
trueState = torch.from_numpy(trueState)
measuredState = torch.from_numpy(measuredState)



if(testFile == 'None'):
    # Generate AR process testing data set - both measured and real state
    testStateData, testStateInfo = ARDatagenMismatch([simu_len, AR_n, AR_var, seq_length], seed+1, args.cuda)
    # Convert the data from normal formatting to batches
    trueStateTEST, measuredStateTEST = convertToBatched(testStateData[2], testStateData[1], batch_size)

    #  #  #  #  #  #
    # See above for why this is commented out
    # # Logging the eval data
    # fileContent[u'evalDataActual'] = trueStateTEST
    # fileContent[u'evalDataMeas'] = measuredStateTEST
    #  #  #  #  #  #
else:


    testDataDict = hdf5s.loadmat(testFile)
    trueStateTEST, measuredStateTEST = convertToBatched(testDataDict['finalStateValues'], testDataDict['observedStateValues'],
                                                batch_size)

    #  #  #  #  #  #
    # measuredStateTEST = testDataDict['measuredData']
    # trueStateTEST = testDataDict['predAndCurState']
    #  #  #  #  #  #

# Convert numpy arrays to tensors
trueStateTEST = torch.from_numpy(trueStateTEST)
measuredStateTEST = torch.from_numpy(measuredStateTEST)

# Generate the model
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ LOAD DATA INTO CUDA ~~~~~~~~~~~~~~~~~~~ ###

if args.cuda:
    torch.cuda.set_device(cuda_device)
    model.cuda()
    trueState = trueState.cuda()
    measuredState = measuredState.cuda()
    trueStateTEST = trueStateTEST.cuda()
    measuredStateTEST = measuredStateTEST.cuda()

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ OPTIMIZER ~~~~~~~~~~~~~~~~~~~~~~~~ ###

# Create the ADAM optimizer
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~ ###

def train(epoch):

    # Initialize training model and parameters
    model.train()
    total_loss = 0

 ################################

    # Training loop - run until we process every series of data
    for i in range(0, measuredState.size(3)):

        # Grab the current series
        x = measuredState[:, :, :, i]
        y = trueState[:, :, i]

        x = x.float()
        y = y.float()

        # Forward/backpass
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
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
                epoch, processed, measuredState.size(3), 100.*processed/measuredState.size(3), lr, cur_loss))
            PredMSE = torch.sum((output[:, 1] - y[:, 1]) ** 2 + (output[:, 3] - y[:, 3]) ** 2) / output.size(0)
            TrueMSE = torch.sum((output[:, 0] - y[:, 0]) ** 2 + (output[:, 2] - y[:, 2]) ** 2) / output.size(0)
            print('PredMSE: ', PredMSE)
            print('TrueMSE: ', TrueMSE)
            total_loss = 0



### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~ EVALUATION ~~~~~~~~~~~~~~~~~~~~~~~~ ###

def evaluate():

    # Total MSE
    TotalAvgPredMSE = 0
    TotalAvgTrueMSE = 0
    n = 0

    # Pre-allocating space for the tensor that will contain the MSEs of each batch from our training set
    testPredMSEs = torch.empty(measuredStateTEST.size(2))
    testEstMSEs = torch.empty(measuredState.size(2))
    # Training loop - run until we only have one batch size of data left
    for i in range(0, (measuredStateTEST.size(2))):

        # Grab the current series
        x_test = measuredStateTEST[:, :, :, i]
        y_test = trueStateTEST[:, :, i]

        x_test = x_test.float()
        y_test = y_test.float()

        # Model evaluation
        model.eval()
        with torch.no_grad():

            # Compute output and loss
            output = model(x_test)
            test_loss = F.mse_loss(output, y_test)

            PredMSE = torch.sum((output[:, 1] - y_test[:, 1]) ** 2 + (output[:, 3] - y_test[:, 3]) ** 2) / output.size(0)
            TrueMSE = torch.sum((output[:, 0] - y_test[:, 0]) ** 2 + (output[:, 2] - y_test[:, 2]) ** 2) / output.size(0)
            testEstMSEs[i] = TrueMSE
            testPredMSEs[i] = PredMSE
            TotalAvgPredMSE+=PredMSE
            TotalAvgTrueMSE+=TrueMSE
        n+=1

    TotalAvgPredMSE = TotalAvgPredMSE / n
    TotalAvgTrueMSE = TotalAvgTrueMSE / n

    predVariance = torch.sum((testPredMSEs[:]-TotalAvgPredMSE)**2)/testPredMSEs.size(0)
    estVariance = torch.sum((testEstMSEs[:]-TotalAvgTrueMSE)**2)/testEstMSEs.size(0)

    # Logging
    fileContent[u'TotalAvgPredMSE'] = repr(TotalAvgPredMSE.item())
    fileContent[u'TotalAvgTrueMSE'] = repr(TotalAvgTrueMSE.item())

    fileContent[u'estimatedVariance'] = estVariance.item()
    fileContent[u'predictedVariance'] = predVariance.item()
    print('TotalAvgPredMSE: ', TotalAvgPredMSE.item())
    print('TotalAvgTrueMSE: ', TotalAvgTrueMSE.item())
    print('Variance of Prediction: ', predVariance.item())
    print('Variance of Estimate: ', estVariance.item())

    return test_loss.item()

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~~~~ LOOP ~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

for ep in range(1, epochs+1):
    train(ep)
    tloss = evaluate()

print("check check")

# End timing
end = time.time()
simRunTime=(end-start)
print('this simulation took:', simRunTime, 'seconds to run')


fileContent[u'trainingLength(seconds)'] = simRunTime
hdf5s.savemat(logName, fileContent)

