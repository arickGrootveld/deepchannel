import hdf5storage as hdf5s
import os
import os.path as path
import numpy as np

def toeplitzifyData(data, **kwargs):
    '''
    toeplitizifyData(data, seqLen=10)
        Parameters:
            data (np.matrix, np.complex128, [N, 1]): Vector of samples from a process to be converted into toeplitz matrix
            seqLen (int): Number of samples in each row of the toeplitz matrix
        Outputs: (toeplitzData)
            toeplitzData (np.matrix, np.complex128, [N-seqLen, seqLen]): A toeplitz matrix create from the data vector
    '''
    seqLen = 10
    if('seqLen' in kwargs):
        seqLen = kwargs['seqLen']

    (_, numSamples) = data.shape
    toeplitzData = np.empty([numSamples - seqLen, seqLen], dtype=complex)

    for i in range(0,numSamples-seqLen):
        toeplitzData[i, :] = data[0, i:i + seqLen]

    return toeplitzData




def matSave(directory, basename, data):
    """
    matSave: Function that saves data to a specified .mat file,
             with the specific file it will be
             saved to being 'directory/basename{#}.mat', 
             where # is the lowest number that will
             not save over another file with the same basename
        Parameters: (directory, basename, data)
            directory (str) - the directory for the data to be 
                              saved in basename (str) - the name of 
                              the file you want the data saved to 
                              before the appended number
            data (dict) - a dict of data to be saved to the .mat file
        Outputs: (logName)
            logName (str) - the name of the file that the 
                            data was saved to
    """
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



def matSaveToFile(filepath, data):
    """
    matSaveToFile(filepath, data): Function to save a dict of data to the filepath
        Parameters: 
            filepath (str): The path where the data is to be saved to. If the path does not end with ".mat",
                            it will be appended onto the filename
            data (dict): The data to be saved. Must be stored in dict form
        Ouputs: null
            This function has no outputs
    """
    ## TODO: Figure out a better way to do this
    if(filepath[-3:] != '.mat'):
        filepath = filepath + '.mat'
    
    hdf5s.savemat(filepath, data)