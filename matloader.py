import hdf5storage as h
import argparse

parser = argparse.ArgumentParser(description='Reading from a specified .mat file - Showing Results stored in .mat files')

# File to read from
parser.add_argument('--filepath', type=str, default='./logs/log1.mat',
                    help='file path to mat file to load from (default: ./logs/log1.mat)')
# Flag to read all values from the .mat file
parser.add_argument('--read_all', action='store_true',
                    help='whether to read all of the data from the file or just the results \
                    (default: False)')
parser.add_argument('--read_keys', action='store_true',
                    help='a flag to tell this program to read only the keys of the dictionary stored in the .mat file (default=False). \
                    This will be overwritten if the --read_all flag is set')
# Entries in dictionary to read from for output
parser.add_argument('--read_values', nargs='+', default=['trainingLength(seconds)', 'TotalAvgPredMSE', 'TotalAvgTrueMSE'],
                    help='entries to read in the dictionary of data stored (default=[\'trainingLength(seconds)\', \'TotalAvgPredMSE\', \'TotalAvgTrueMSE\'].\
                    This will be overwritten if the --read_all or --read_keys flags are set')

args = parser.parse_args()

matFile = args.filepath

fileContents = h.loadmat(matFile)


if(args.read_all):
    print(fileContents)
elif(args.read_keys):
    print(fileContents.keys())
else:
    printList = []
    for entry in args.read_values:
        printList.append(fileContents[entry])
    print(printList)
