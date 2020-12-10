# Gilbert Elliot Simulation Directory

This directory holds all the tools needed to recreate the Gilbert Elliot results shown in the paper


## Files of interest
- gilbertElliotSimulation.py
- gilbertElliotDataGen.py

These two files compromise all the GE functionality in this repo. *gilbertElliotDataGen.py* will generate data from a corresponding GE simulation. 
*gilberElliotSimulation.py* will train a TCN and a LS filter, and then compare both of their performances against a Kalman Filter and each other. 
For a more comprehensive understanding of the functionality of each of these scripts, read through the comments and use the --help command for their
CLI's.

## Example usage
Example data files have already been generated, and an example model and log file also exist. If you would like to see what running the model looks like, try running
```
python gilbertElliotSimulation.py --testDataFile=data/GETestDataExample.mat --trainDataFile=data/GETrainDataExample.mat --evalDataFile=data/GEEvalDataExample.mat
```

This will train a simple model on a very limited set of data. This will also default to using the CPU on your system, which may take a significant amount of time, 
or cause other issues. 

If you have Cuda installed, and have a GPU available, it is far more beneficial to run the program such that it will use all available GPU's, such as with this command
```
python gilbertElliotSimulation.py --testDataFile=data/GETestDataExample.mat --trainDataFile=data/GETrainDataExample.mat --evalDataFile=data/GEEvalDataExample.mat --cuda
```

If you would like to generate a more sizeable amount of data to train, or evaluate your model, you can do so with the following command (you will want to increase the simu_len parameter as you see fit to generate additional data)
```
python gilbertElliotDataGen.py --simu_len=1000 --seq_len=20
```

If you want to generate a larger sample of test data, you can do so with this command (again, changing the simu_len parameter as you see fit)
```
python gilbertElliotDataGen.py --simu_len=1000 --seq_len=20 --testDataGen
```
