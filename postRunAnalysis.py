import hdf5storage as hdf5s
import argparse

# parseOutDataFilePat: Parses out the base path that a run took place from,
#                              based on the logPath, and uses that to generate the
#                              path to the data file specified
# Inputs: (logFilePath, dataFilePath)
#   logFilePath (str) - the path that a log file was loaded from
#   dataFilePath (str) - the path from where the run was generated that the
#                        data file was saved to
# Outputs: (realDataFilePath)
#   realDataFilePath (str) - the path from where the log file was loaded to the actual data file
def parseOutDataFilePath(logFilePath, dataFilePath):
    # Grabbing the file path that precedes the logs file
    baseFilePath = logFilePath.split('logs')[0]
    realDataFilePath = baseFilePath + dataFilePath

    return(realDataFilePath)


# Create argument parser
parser = argparse.ArgumentParser(description='Post Run Analysis - Organizes data generated by a simulation run of '
                                             'the TCN, KF, or LeastSquares methods')

# Path to the log file
# Will determine what type of run it was based on a entry in the log file that specifies its type
parser.add_argument('--logFilePath', type=str, default='None',
                    help='The path to the log file to analyze, will automatically determine what kind of '
                         'run it was based on the log file (default=None')


args = parser.parse_args()

if not (args.logFilePath ==  'None'):
    logFile = hdf5s.loadmat(args.logFilePath)
    print('loaded log file at: ', logFile)
else:
    raise Exception('Must give a log file in order to run analysis')

# Determining what kind of run the log file is from
runType = logFile['runType']

if(runType == 'TCN'):
    dataFilePath = logFile['testDataFile']
    estMSE = logFile['TotalAvgTrueMSE']
    predMSE = logFile['TotalAvgPredMSE']

elif(runType == 'LS'):
    dataFilePath = logFile['dataFilePath']
    estMSE = logFile['MSE_est']
    predMSE = logFile['MSE_pred']

elif(runType == 'KF'):
    dataFilePath = logFile['dataFileName']
    estMSE = logFile['predictionMSE']
    predMSE = logFile['estimatedMSE']
else:
    raise Exception('log file did not specify run type, pleas double check this')

# Loading the data file and grabbing the values that the riccati equations converge to
actualDataFilePath = parseOutDataFilePath(args.logFilePath, dataFilePath)
dataFile = hdf5s.loadmat(actualDataFilePath)
print('loaded data from: ', actualDataFilePath)
riccatiMSEs = dataFile['riccatiConvergences']
riccatiPredictionMSEs = riccatiMSEs[0,:]
riccatiEstimateMSEs = riccatiMSEs[1,:]

runData = {}

runData[u'riccatiPredictionMSEs'] = riccatiPredictionMSEs
runData[u'riccatiEstimateMSEs'] = riccatiEstimateMSEs
runData['actualPredictionMSE'] = predMSE
runData['actualEstimateMSE'] = estMSE

# TODO: Find a format for the data so you can have testing happen across a batch of AR Processes that each have their
# TODO: own AR Coeffecients


print('thank you for your time')






