"""
GaussMarkovProcess: Provides functions to generate samples from a Gauss Markov Process
"""

import numpy as np
import scipy.linalg as sciAlg

def GaussMarkovProcess(**kwargs):
    '''
    GaussMarkovProcess(simuLen=100, q_covar=0.1, r_covar=0.1, ar_coeffs=(0.3, 0.1), seed=-1)
        Parameters:
            simuLen (int): The number of samples to generate from the Gauss Markov Process
            q_covar (float): The covariance value of the process noise
            r_covar (float): The covariance value of the observation noise
            ar_coeffs (list/tuple): The coeffecients of the transition matrix of the GA Process
            seed (int): The seed for the rng of this process, if not set it will default to a random seed
        Outputs: (x, z, ricattiPredConvergence, riccatiEstConvergence)
            x (complex128 np.matrix [simuLen, 1]): A vector of the process states from the GA Process
            z (complex128 np.matrix [simuLen, 1]): A vector of the observation states from the GA Process
            riccatiPredConvergence (float): The value that the Riccati Equation claims the MMSE
                                            Filter would converge to for predicting the state
            riccatiEstConvergence (float): The value that the Riccati Equation claims the MMSE
                                           Filter would converge to for estimating the state
    '''

    simuLen = 100
    q_covar = 0.1
    r_covar = 0.1
    ar_coeffs = (0.3, 0.1)
    seed = -1

    if('simuLen' in kwargs):
        simuLen = kwargs['simuLen']
    if('ar_coeffs' in kwargs):
        ar_coeffs = kwargs['ar_coeffs']
    if('q_covar' in kwargs):
        q_covar = kwargs['q_covar']
    if('r_covar' in kwargs):
        r_covar = kwargs['r_covar']
    if('seed' in kwargs):
        seed = kwargs['seed']


    F = np.matrix([[ar_coeffs[0], ar_coeffs[1]], [1, 0]], dtype=float)

    # Process Noise Covariance Matrix
    Q = np.matrix([[q_covar, 0], [0, 0]], dtype=float)

    # Observation noise covariance matrix
    R = np.matrix(r_covar, dtype=float)

    # Matrix mapping real states into observation domain
    H = np.matrix([1,0], dtype=float)

    # Pre-allocating process states
    x = np.empty([1, simuLen], dtype=complex)

    # Pre-allocating the stored observed states
    z = np.empty([1, simuLen], dtype=complex)

    # Calculating the Riccati Convergences of this GA Process
    riccatiPred = sciAlg.solve_discrete_are(np.transpose(F), np.transpose(H), Q, R)

    kRicConIntermediate = np.add(np.matmul(np.matmul(H, riccatiPred), np.transpose(H)), R)
    riccatiKalGain = np.matmul(np.matmul(riccatiPred, np.transpose(H)), np.linalg.inv(kRicConIntermediate))

    riccatiEst = riccatiPred - (np.matmul(np.matmul(riccatiKalGain, H), riccatiPred))

    print(riccatiPred)
    print(riccatiEst)

    
    initStates = (0,0)
    for i in range(0, simuLen):
        
        (x_curr, z_curr) = GaussMarkovSample(init_states=initStates, q_covar=q_covar,
                                             r_covar=r_covar, ar_coeffs=ar_coeffs, seed=seed)
        x[0, i] = x_curr
        z[0, i] = z_curr
        initStates = (x_curr, initStates[0])

    return x, z, riccatiPred, riccatiEst






def GaussMarkovSample(**kwargs):
    '''
    GaussMarkovSample(initStates=(0,0), q_covar=0.1, r_covar=0.1, ar_coeffs=(0.3, 0.1), seed=-1)
        Parameters:
            initStates (list/tuple): The states of the process at k-1 and k-2 respectfully
            q_covar (float): The covariance value of the process noise
            r_covar (float): The covariance value of the observation noise
            ar_coeffs (list/tuple): The coeffecients of the transition matrix of the GA Process
            seed (int): The seed for the rng of this process, if not set it will default to a random seed
        Outputs: (x, z)
            x (np.complex128): The current (k) state of the process
            z (np.complex128): The current (k) observation of the process
    '''
    # Setting up the default values of the function
    initStates = (0,0)
    ar_coeffs = (0.3, 0.1)
    q_covar = 0.1
    r_covar = 0.1
    seed = -1

    if('initStates' in kwargs):
        initStates = kwargs['initStates']
    if('ar_coeffs' in kwargs):
        ar_coeffs = kwargs['ar_coeffs']
    if('q_covar' in kwargs):
        q_covar = kwargs['q_covar']
    if('r_covar' in kwargs):
        r_covar = kwargs['r_covar']
    if('seed' in kwargs):
        seed = kwargs['seed']

    # If seed specified then use it, otherwise use random seed
    if seed > 0:
        np.random.seed(seed)
    # No else statement needed, if seed not provided to numpy,
    # then it will use random seed

    # Assigning AR Process matrices from input parameters
    F = np.matrix([ar_coeffs, [0, 1]])
    Q_Chol = np.matrix([[np.sqrt(q_covar), 0], [0, 0]])
    R = np.matrix(r_covar)

    x_prev = np.transpose(np.matrix([initStates[0], initStates[1]]))

    # Defining preset matrices
    H = np.matrix([1,0])


    # Starting the AR Process here
    rProcessNoise = np.divide(np.matmul(Q_Chol, np.random.randn(2,1)), np.sqrt(2))
    iProcessNoise = 1j * np.divide(np.matmul(Q_Chol, np.random.randn(2,1)), np.sqrt(2))

    v = rProcessNoise + iProcessNoise

    rObsNoise = np.divide(np.matmul(np.sqrt(R), np.random.randn(1)), np.sqrt(2))
    iObsNoise = 1j * np.divide(np.matmul(np.sqrt(R), np.random.randn(1)), np.sqrt(2))

    w = rObsNoise + iObsNoise

    x = np.matmul(F, x_prev) + v
    z = np.matmul(H,x) + w
    x = x[0]

    return x, z