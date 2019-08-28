import argparse
import time

from mismatch_data_gen import ARDatagenMismatch

parser = argparse.ArgumentParser(description='Generate data for testing purposes. For use with the Kalman Filter mismatch code,\
                                             and the TCN')
# Number of batches of data
parser.add_argument('--simLength', type=float, default=10,
                    help='the number of batches of data to generate (default=10)')

# Sequence Length of each batch element
parser.add_argument('--sequenceLength', type=float, default=20,
                    help='number of samples in each AR process that makes up a batch element (default=20)')

# Variance of AR Coeffieients
parser.add_argument('--arVar', type=float, default=0,
                    help='the variance of the AR Coefficients for each AR process (default=0)')

# Order of AR process
parser.add_argument('--AR_n', type=int, default=2,
                    help='the order of the AR process that will be generated (default=2)' \
                         '{as of this comment, features related to this parameter have not been implmented yet, so' \
                         'change the value of this from the default at your own peril}')
# GPU used for data generation
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA for data generation (default: False)')

# Seed for the random number generation
parser.add_argument('--seed', type=int, default=1111,
                    help='seed for the rng of the data')

args = parser.parse_args()

# Setting up a time to determine how long data generation takes
start = time.time()

stateData, info = ARDatagenMismatch([int(args.simLength),args.AR_n, args.arVar, int(args.sequenceLength)],
                                             args.seed, args.cuda)
end=time.time()

print('it took this many seconds to generate the data', end-start)

print(args)

