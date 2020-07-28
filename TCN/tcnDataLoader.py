import torch as t
import hdf5storage as hdf5
from torch.utils.data import DataLoader, IterableDataset

class matTrainFileIterator(IterableDataset):
    def __init__(self, dataFile):
        super(IterableDataset).__init__()

        self.dataFile = dataFile
    
        self.currentSample = 0
        print(self.dataFile)
    
    def __iter__(self):
        print('this is an iteration')
        
        # Getting the worker id to allow us to split the data up
        worker_info = t.utils.data.get_worker_info()
        
        #if worker_info is None:

        
        test = hdf5.loadmat(self.dataFile)
        print(type(test))
        print(test.keys())
        xVals = test['observedStates']
        yVals = test['finalStateValues']
        self.currentSample = self.currentSample + 1
        print(self.currentSample)
        return(iter([1,2,3,4]))

# Testing bed for data loader

trainData = matTrainFileIterator('../data/GEData212.mat')

trainLoaderDummy = DataLoader(trainData, batch_size=1)

print(list(trainLoaderDummy))

