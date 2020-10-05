import hdf5storage as h5
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Concatenate data files together to get around matlabs stupid formatting issues")

parser.add_argument('--dataFiles', metavar='F', type=str, nargs='+',
                    help='the data files to be collected and concatenated')

parser.add_argument('--saveFile', metavar='S', type=str, default='combinedData.mat',
                    help='where to save the concatenated data (default: combinedData.mat)')
args = parser.parse_args()

dataList = []
finalDataShape = 0

for i in args.dataFiles:
    print("loading data from {file}".format(file=i))
    targData = h5.loadmat(i)
    dataList.append(targData)
    finalDataShape = finalDataShape + targData['XX'].shape[0]

        
dataXX = np.zeros((finalDataShape, dataList[0]['XX'].shape[1],2))
dataYY = np.zeros((finalDataShape, 2))

currentLoc = 0
for m in dataList:
    dataXX[currentLoc:(m['XX'].shape[0] + currentLoc), :, :] = m['XX'][:, :, :]
    dataYY[currentLoc:(m['YY'].shape[0] + currentLoc), :] = m['YY'][:, :]

    currentLoc = currentLoc + m['XX'].shape[0]


saveData = dict()

saveData['XX'] = dataXX
saveData['YY'] = dataYY

print("saving data to {file}".format(file=args.saveFile))
h5.savemat(args.saveFile, saveData)

