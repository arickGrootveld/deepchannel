import hdf5storage as h
import argparse

parser = argparse.ArgumentParser(description='Printing the results of the neural network from the mat file specified - Showing Results of TCN Run')

# File to read from
parser.add_argument('--file', type=str, default='log6.mat',
                    help='mat file to load from (default: log6.mat)')

parser.add_argument('--read_all', action='store_true',
                    help='whether to read all of the data from the file or just the results \
                    (default: False)')

args = parser.parse_args()

matFile = './logs/'+ args.file

fileContents = h.loadmat(matFile)


if(args.read_all):
    print(fileContents)
else:
    print('estimate Avg MSE: ', fileContents['TotalAvgTrueMSE'])
    print('predicted Avg MSE: ', fileContents['TotalAvgPredMSE'])
    print('run time (seconds): ', fileContents['trainingLength(seconds)'])

