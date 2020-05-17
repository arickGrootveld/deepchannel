'''
GilbertElliotProcess: a file for generating data from a Gilbert Elliot Channel Model
'''
# If this file is the main file then we need to add the right path to the sys path
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath('.')) # Adds the base path that the
                                      # file is run from to the
                                      # system path

import numpy as np
from datagen.GaussMarkovProcess import GaussMarkovSample

def gilbertElliotProcess(**kwargs):
    '''
    gilbertElliotProcess(simuLen=100, goodCoeffs=(0.3, 0.1), badCoeffs=(1.949, -0.95),
                         transProb=0.0005, initialState=-1, seed=-1)
        Parameters:
            simuLen (int): Number of samples of Gilbert Elliot Channel Model to generate
            goodCoeffs (list/tuple): Coefficients of the good channel transition matrix
            badCoeffs (list/tuple): Coefficients of the bad channel transition matrix
            transProb (float): Probability of transitifoning from one channel to other, should be below 1
            initialState (int): Which channel to start the process in. If set to 0, will start in
                                good channel. If set to 1, will start in bad channel. If set
                                to -1 will start in random channel
            seed (int): The seed for the rng of this process, if not set it will default to a random seed
        Outputs: (x, z, channelState)
            x (np.matrix, np.complex128, [simuLen, 1]): A vector of the process state values from GE Model
            z (np.matrix, np.complex128, [simuLen, 1]): A vector of the observed state values from GE Model
            channelState (np.matrix, [simuLen, 1]): A vector of values tracking which channel the 
                                                    GE Model was in for each sample. 1's correspond to
                                                    the bad channel, 0's correspond to the good channel
    '''

    # Initializing variables
    simuLen = 100
    goodCoeffs = (0.3, 0.1)
    badCoeffs = (1.949, -0.95)
    transProb = 0.0005
    initialState = -1
    seed=-1

    # Deconstructing the kwargs
    if('simuLen' in kwargs):
        simuLen = kwargs['simuLen']
    if('goodCoeffs' in kwargs):
        goodCoeffs = kwargs['goodCoeffs']
    if('badCoeffs' in kwargs):
        badCoeffs = kwargs['badCoeffs']
    if('transProb' in kwargs):
        transProb = kwargs['transProb']
    if('initialState' in kwargs):
        initialState = kwargs['initialState']
    if('seed' in kwargs):
        seed = kwargs['seed']

    # Initializing the seed with a random value if no seed provided
    if(seed <= 0):
        seed = np.random.randint(1)

    # Initializing the markov chain
    transProbArray = ((1-transProb, transProb), (1-transProb, transProb))
    transStArray = (('good', 'bad'), ('bad', 'good'))

    if(initialState == 0):
        MarkovState = 'good'
    elif(initialState == 1):
        MarkovState = 'bad'
    else:
        startingStates = ('good', 'bad')
        startProbs = (0.5, 0.5)
        MarkovState = np.random.choice(startingStates, replace=True, p=startProbs)
    

    # Initializing the Gauss Markov Process that is the underlying model
    # of this Gilbert Elliot Process

    # Preallocating the process states
    x = np.empty([1,simuLen], dtype=complex)
    # Assigning a variable to the states used for computing the next sample
    x_current = [0, 0]

    # Preallocating the observed states
    z = np.empty([1, simuLen], dtype=complex)

    # Setting up the channel state memory
    channelState = np.empty([1, simuLen])

    # Main loop to generate the sequence of data
    for i in range(0,simuLen):
        
        # Determining the current channel state
        if(MarkovState == 'good'):
            currentCoeffs = goodCoeffs
            channelState[0, i] = 0
        elif(MarkovState == 'bad'):
            currentCoeffs = badCoeffs
            channelState[0, i] = 1
        else:
            raise Exception('Markov Chain got into unintended state')

        # Getting the current process and observation states
        [x_val, z_val] = GaussMarkovSample(initStates=x_current, \
                                           ar_coeffs=currentCoeffs, \
                                           seed=seed+i)
        x_current = [x_val.item(), x_current[0]]
        z[0, i] = z_val
        x[0, i] = x_val

        # Determining the next channel state
        if(MarkovState == 'good'):
            MarkovState = np.random.choice(transStArray[0], replace=True, \
                                           p=transProbArray[0])
        elif(MarkovState == 'bad'):
            MarkovState = np.random.choice(transStArray[1], replace=True, \
                                           p=transProbArray[1])
        else:
            raise Exception('Markov chain got into unitended state')
    
    return (x, z, channelState)



if __name__ == "__main__":
    """
    If this file is being executed, then we want to generate a 
    """
    
    from datagen.utilities import matSave, toeplitzifyData, matSaveToFile
    import argparse
    from os import path
    
    import time
    startTime = time.time()

    parser = argparse.ArgumentParser(description='Generating a toeplitz matrix of process and observed states' + 
    'from a Gilbert Elliot Process')
    
    parser.add_argument('--numSamples', type=int, default=1000,
                        help='number of samples of process to generate (default: 1000)')
    
    parser.add_argument('--seqLen', type=int, default=10,
                        help='length of the rows of the toeplitz matrix (default: 10)')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')

    parser.add_argument('--goodCoeffs', nargs='+', default=[0.3, 0.1],
                        help='Coefficients for the good channel of the GE Process (default: [0.3, 0.1])')

    parser.add_argument('--badCoeffs', nargs='+', default=[1.949, -0.95],
                        help='Coefficients for the bad channel of the GE Process (default: [1.949, -0.95])')
    
    parser.add_argument('--saveLoc', type=str, default='data/GEData',
                        help='Directory to save the data generated (default: data/GEData)')

    # Parse out the input arguments
    args = parser.parse_args()

    # Assigning pre run parameters
    saveLoc = os.path.abspath('.') + '/' + args.saveLoc
    numSamples = args.numSamples
    seqLen = args.seqLen
    goodCoeffs = args.goodCoeffs
    badCoeffs = args.badCoeffs
    seed = args.seed

    # Saving a blank .mat file to the file that the data will eventually be saved to, 
    # that if two process are running at the same time, they won't save over each other
    saveData = dict()
    saveFilepath = matSave(saveLoc, 'GEData', saveData)

    [prosStates, obsStates, _] = gilbertElliotProcess(simuLen=numSamples, goodCoeffs=goodCoeffs,
                                                      badCoeffs=badCoeffs, seed=seed)
    prosStates = toeplitzifyData(prosStates, seqLen=seqLen)
    obsStates = toeplitzifyData(obsStates, seqLen=seqLen)

    saveData['processStates'] = prosStates
    saveData['observedStates'] = obsStates

    saveData['bagCoeffs'] = badCoeffs
    saveData['goodCoeffs'] = goodCoeffs
    saveData['seqLen'] = seqLen
    saveData['numSamples'] = numSamples
    saveData['seed'] = seed

    finishTime = time.time()

    timeTaken = finishTime - startTime

    saveData['timeTaken'] = timeTaken
    print('It took ' + str(timeTaken) + ' seconds to generate the data')
    
    matSaveToFile(saveFilepath, saveData)








        
