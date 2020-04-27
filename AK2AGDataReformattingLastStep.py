import hdf5storage as h
import pandas as pd
import numpy as np

test = h.loadmat('data/GETestDataAK4.mat')
if(False):
    recreatedData = []
    targetDataName = 'test3'
    targetData = test[targetDataName]
    test1 = pd.DataFrame.from_dict(targetData[0])
    test2 = {}
    for i in test1.keys():
        test2[i] = test1[i][0][0][0]
    print(test2)


CT =  test['testDataInfo']
df = pd.DataFrame.from_dict(CT[0])

# Reformatting the testDataInfo to be in a dict so it can be loaded properly
potentialFormat = []
for j in range(0,len(df)):
    potDict = {}
    for i in df.keys():
        potDict[i] = df[i][j][0]
        if(potDict[i].dtype.type != np.str_):
                potDict[i] = potDict[i][0]
    potentialFormat.append(potDict)
test['testDataInfo'] = potentialFormat

# Reformatting LSandKFTestData to be in a list of arrays so it can be used
LSandKFTestDataReformatted = []
unformattedLSandKFTestData = test['LSandKFTestData']
# Getting number of variables stored; there are the # of runs*3 variables
_,n = unformattedLSandKFTestData.shape

if(n%3 == 0):
    LSaKFTDSubset = []
    for i in range(0,n):
        LSaKFTDSubset.append(unformattedLSandKFTestData[0][i])
        # Every 3rd variable is the last variable from that data set, and so
        # we append the data, and start a new array for the next data set
        if((i+1)%3 == 0):
            LSandKFTestDataReformatted.append(LSaKFTDSubset)
            LSaKFTDSubset = []

test['LSandKFTestData'] = LSandKFTestDataReformatted

h.savemat('data/GETestDataAK5.mat', test)

