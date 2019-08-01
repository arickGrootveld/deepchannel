# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#            TCN Based Channel Estimation Neural Network Training
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import os
from os import path

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from model import TCN
from data_gen import ARdatagen

# Timer for logging how long the training takes to execute
import time
start = time.time()

# Importing file handler for .mat files
import hdf5storage as hdf5s

### ~~~~~~~~~~~~~~~~~~~~ ARGUMENT PARSER ~~~~~~~~~~~~~~~~~~~~ ###
### Argument parser for easier and cleaner user interface

# Create argument parser
parser = argparse.ArgumentParser(description='Sequence Modeling - Complex Channel Gain Estimation')

# Batch size
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')

# CUDA enable
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')

# CUDA device
parser.add_argument('--cuda_device', type=int, default=0,
                    help='cuda device to use if running on cuda, only works if cuda \
                    is enabled (default 0)')

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
parser.add_argument('--log-interval', type=int, default=5000, metavar='N',
                    help='report interval (default: 5000')

# Learning rate
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate (default: 2e-4)')

# Optimizer
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')

# Hidden units per layer
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')

# Random seed value
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

# Data Generation Length
parser.add_argument('--simu_len', type=float, default=50000,
                    help='amount of data generated for training (default: 50000)')

# Length of data used for Evaluation of models effectiveness
parser.add_argument('--test_len', type=float, default=1e6,
                    help='amount of data generated for testing (default: 1e6)')


# Format of data input to model as either complex and real or angle and magnitude
parser.add_argument('--polar', action='store_true',
                    help='data from AR process in polar format (default: False)')

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
#logName='./logs/log' + str(logNumber) + '.mat'

# Logging the parameters of the model
print('model parameters: ', args)
fileContent = {}
# First write
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
data_polar = args.polar
testDataLen = int(args.test_len)
cuda_device = args.cuda_device

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~ LOAD DATA/GENERATE MODEL ~~~~~~~~~~~~~~~~ ###

# ~~~~~~~~~~~~~~~~~~ LOAD TRAINING SET

# Real numbers on the left column, imaginary on the right column
# Generate AR process data - both measured and real state
stateData, measuredData = ARdatagen(2, simu_len, seed, data_polar)

# Logging the train data
fileContent[u'trainDataMeas'] = measuredData
fileContent[u'trainDataActual'] = stateData

##########################
# Load data from text files if needed
##dataReal = np.loadtxt('realData.txt', delimiter=',')
##dataObserved = np.loadtxt('observedData.txt', delimiter=',')
##########################
# Convert numpy arrays to tensors
stateData = torch.from_numpy(stateData)
measuredData = torch.from_numpy(measuredData)

# Resize tensors
stateData = stateData.t()
measuredData = measuredData.t()

stateData = stateData.unsqueeze(1)
measuredData = measuredData.unsqueeze(1)

stateData = torch.reshape(stateData, (1, 2, -1))
measuredData = torch.reshape(measuredData, (1, 2, -1))


# ~~~~~~~~~~~~~~~~~~ LOAD TEST SET
# Real numbers on the left column, imaginary on the right column
# Generate AR process data - both measured and real state
stateDataTEST, measuredDataTEST = ARdatagen(2, testDataLen, seed+1, data_polar)

# Logging the eval data
fileContent[u'evalDataMeas'] = measuredDataTEST
fileContent[u'evalDataActual'] = stateDataTEST

##########################
# Load data from text files if needed
##dataReal = np.loadtxt('realData.txt', delimiter=',')
##dataObserved = np.loadtxt('observedData.txt', delimiter=',')
##########################

# Convert numpy arrays to tensors
stateDataTEST = torch.from_numpy(stateDataTEST)
measuredDataTEST = torch.from_numpy(measuredDataTEST)

# Resize tensors
stateDataTEST = stateDataTEST.t()
measuredDataTEST = measuredDataTEST.t()

stateDataTEST = stateDataTEST.unsqueeze(1)
measuredDataTEST = measuredDataTEST.unsqueeze(1)

stateDataTEST = torch.reshape(stateDataTEST, (1, 2, -1))
measuredDataTEST = torch.reshape(measuredDataTEST, (1, 2, -1))

# Generate the model
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~ LOAD DATA INTO CUDA ~~~~~~~~~~~~~~~~~~~ ###

if args.cuda:
    torch.cuda.device(cuda_device)
    model.cuda()
    stateData = stateData.cuda()
    measuredData = measuredData.cuda()
    stateDataTEST = stateDataTEST.cuda()
    measuredDataTEST = measuredDataTEST.cuda()

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ OPTIMIZER ~~~~~~~~~~~~~~~~~~~~~~~~ ###

# Create the ADAM optimizer
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
### ~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~ ###


## TRAINING LOOP MUST BE CHANGED TO WORK WITH CURRENT DATA GEN
## WHICH OUTPUTS A 4D TENSOR OF TOEPLITZ TENSORS 

def train(epoch):

    # Initialize training model and parameters
    global lr1
    model.train()
    total_loss = 0

################################

    # Training loop - run until we only have one batch size of data left
    for i in range(0, (measuredData.size(2))-(seq_length*batch_size)):

        # Get the first segment
        x = measuredData[:, :, i:i+seq_length]
        y = stateData[:, :, i+seq_length-1:i+seq_length+1]

        # Construct data (toeplitz) matrix
        for j in range(i+1, batch_size+i):
            x = torch.cat((x, measuredData[:, :, j:j+seq_length]), 0)
            y = torch.cat((y, stateData[:, :, j+seq_length-1:j+seq_length+1]), 0)


        x = x.float()
        y = y.float()

        # Resize
        y = torch.reshape(y, (32, 4))

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

        # Display training results
        if i % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, measuredData.size(2))

            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, measuredData.size(2), 100.*processed/measuredData.size(2), lr, cur_loss))

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

    # Training loop - run until we only have one batch size of data left
    for i in range(0, (measuredDataTEST.size(2)) - (seq_length * batch_size)):

        # Get the first segment
        x_test = measuredDataTEST[:, :, i:i+seq_length]
        y_test = stateDataTEST[:, :, i+seq_length-1:i+seq_length+1]

        # Construct data (toeplitz) matrix
        for j in range(i+1, batch_size+i):
            x_test = torch.cat((x_test, measuredDataTEST[:, :, j:j+seq_length]), 0)
            y_test = torch.cat((y_test, stateDataTEST[:, :, j+seq_length-1:j+seq_length+1]), 0)

        # reshaping
        x_test = x_test.float()
        y_test = y_test.float()
        y_test = y_test.squeeze()
        y_test = torch.reshape(y_test, (32, 4))

        # Model evaluation
        model.eval()
        with torch.no_grad():

            # Compute output and loss
            output = model(x_test)
            test_loss = F.mse_loss(output, y_test)

            PredMSE = torch.sum((output[:, 1] - y_test[:, 1]) ** 2 + (output[:, 3] - y_test[:, 3]) ** 2) / output.size(0)
            TrueMSE = torch.sum((output[:, 0] - y_test[:, 0]) ** 2 + (output[:, 2] - y_test[:, 2]) ** 2) / output.size(0)
            TotalAvgPredMSE+=PredMSE
            TotalAvgTrueMSE+=TrueMSE
        n+=1

    # Print average MSE over entire test set (true and predicted)
    TotalAvgPredMSE = TotalAvgPredMSE / n
    TotalAvgTrueMSE = TotalAvgTrueMSE / n

    # Logging
    fileContent[u'TotalAvgPredMSE'] = repr(TotalAvgPredMSE.item())
    fileContent[u'TotalAvgTrueMSE'] = repr(TotalAvgTrueMSE.item())
    print('TotalAvgPredMSE: ', TotalAvgPredMSE.item())
    print('TotalAvgTrueMSE: ', TotalAvgTrueMSE.item())

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




fileContent[u'trainingLength(seconds)'] = simRunTime
hdf5s.savemat(logName, fileContent)

