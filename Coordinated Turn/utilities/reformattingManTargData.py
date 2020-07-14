import hdf5storage as h5
import argparse
import os

parser = argparse.ArgumentParser(description="""Reformatting .mat data 
                                                generated
                                                by matlab to work
                                                with our TCN code""")
parser.add_argument('--targFile', type=str, default='None', metavar='t',
                    help='The file to reformat and save over')

parser.add_argument('--testData', action='store_true',
                    help='convert to test data (default: False)')


args = parser.parse_args()

targetFile = args.targFile

print('Loading data from: ', targetFile)

matlabFormattedData = h5.loadmat(targetFile)

runData = dict()

# If the data is already reformatted, then we can simply inform the user
# that this is the case and leave the data alone
if 'reformattedData' in matlabFormattedData.keys():
    runData = matlabFormattedData
    print('data in: ' + targetFile + ' is already reformatted')
else:

    os.remove(targetFile)

    # If we are generating test data, then we need to have a slightly
    # different format for the data
    if(args.testData):
        runData['reformattedData'] = True

        # Rewriting the data to a test data file
        inter1 = targetFile.rfind('/')
        inter2 = targetFile.find('ManTargData')
        inter3 = targetFile.find('.')
        targetFile = targetFile[0:inter1 + 1] + 'ManTargTestData' \
            + targetFile[inter2 + 11:inter3] + '.mat'
        
        LSAndKFTestData = matlabFormattedData['data'][0,0]['systemStates']
        runData['LSAndKFTestData'] = [LSAndKFTestData]

        runData['measuredStateTEST'] = [matlabFormattedData['data'][0,0]['observedStates']]

        runData['trueStateTEST '] = [matlabFormattedData['data'][0,0]['finalStateValues']]

        inter1 = dict()
        inter1['riccatiConvergenceEst'] = 0.00
        inter1['riccatiConvergencePred'] = 0.00
        inter1['seed'] = 0
        runData['testDataInfo'] = [inter1]
    else:
        runData['seed'] = matlabFormattedData['data'][0,0]['seed'][0][0]
        runData['finalStateValues'] = matlabFormattedData['data'][0,0]['finalStateValues']
        runData['observedStates'] = matlabFormattedData['data'][0,0]['observedStates']
        runData['systemStates'] = matlabFormattedData['data'][0,0]['systemStates']

        # Intermediate variable to hold the parameters so they can be 
        # reorganized in usable fashion
        intermed = dict()
        intermed['numSequences'] = matlabFormattedData['data'][0,0]['parameters'][0]['numSequences'][0][0][0]
        intermed['sequenceLength'] = matlabFormattedData['data'][0,0]['parameters'][0]['sequenceLength'][0][0][0]
        # Saving the parameters to data to be saved
        runData['parameters'] = intermed

        runData['riccatiConvergences'] = matlabFormattedData['data'][0,0]['riccatiConvergences']
        runData['reformattedData'] = True
    print('saving data to: ' + targetFile)

    # Saving the data to its new file
    h5.savemat(targetFile, runData)





