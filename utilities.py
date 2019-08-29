import hdf5storage as hdf5s
import os
import os.path as path
import numpy as np

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

