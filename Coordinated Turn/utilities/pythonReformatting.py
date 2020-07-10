import hdf5storage as h5
import argparse
import os

parser = argparse.ArgumentParser(description="""Reformatting .mat data 
                                                generated
                                                by matlab to work
                                                with our TCN code""")
parser.add_argument('--targFile', type=str, default='None', metavar='t',
                    help='The file to reformat and save over')


args = parser.parse_args()

targetFile = args.targFile

print('Loading data from: ', targetFile)

matlabFormattedData = h5.loadmat(targetFile)

runData = dict()

if 'reformattedData' in matlabFormattedData.keys():
    runData = matlabFormattedData
    print('data in: ' + targetFile + ' is already reformatted')
else:
    runData['seed'] = matlabFormattedData['data'][0,0]['seed'][0][0]
    runData['finalStateValues'] = matlabFormattedData['data'][0,0]['finalStateValues']
    runData['observedStates'] = matlabFormattedData['data'][0,0]['observedStates']

    # Intermediate variable to hold the parameters so they can be 
    # reorganized in usable fashion
    intermed = dict()
    intermed['numSequences'] = matlabFormattedData['data'][0,0]['parameters'][0]['numSequences'][0][0][0]
    intermed['sequenceLength'] = matlabFormattedData['data'][0,0]['parameters'][0]['sequenceLength'][0][0][0]
    # Saving the parameters to data to be saved
    runData['parameters'] = intermed

    runData['riccatiConvergences'] = matlabFormattedData['data'][0,0]['riccatiConvergences']
    runData['reformattedData'] = True

os.remove(targetFile)

h5.savemat(targetFile, runData)


