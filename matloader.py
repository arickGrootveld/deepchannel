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
                    This will be overwritten if the --read_all, --read_keys, or --read_size flags are set')

# Read the size of the data that the filepath is pointed to
parser.add_argument('--read_size', action='store_true',
                    help='Read back the size of the data that the filepath points to. Will return how many sequences are stored in'
                         'a specific file. Will be overwritten if the --read_all, or --read_keys flags are set')

args = parser.parse_args()

matFile = args.filepath

fileContents = h.loadmat(matFile)


if(args.read_all):
    print(fileContents)
elif(args.read_keys):
    print(fileContents.keys())
elif(args.read_size):
    if('finalStateValues' in fileContents.keys()):
        dataSize = fileContents['finalStateValues'].shape[1]
    elif('trueStateTEST' in fileContents.keys()):
        dataShape = fileContents['trueStateTEST'].shape
        dataSize = (dataShape[0], dataShape[3])
    else:
        raise Exception('{} is not a data file, please use a file with correct keys'.format(matFile))
    print('there are {} sequences stored in this file'.format(dataSize))
else:
    printList = []
    for entry in args.read_values:
        printList.append(fileContents[entry])
    print(printList)
