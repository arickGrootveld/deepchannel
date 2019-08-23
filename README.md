# DATA GENERATION

To generate AR process data, the following python scripts are involved:

* mismatch_data_gen.py
* generateTestData.py


### mistmatch_data_gen.py

This script handles the generation of the AR process data. This script is only explicitly called inside
of the neural network or in the Kalman filter code if there is no data set provided via a command line argument
(no valid path to a .mat file). If you wish to generate a dataset using this script, use the generateTestData.py wrapper 
script to save data to a .mat file which you can then load in subsequent programs via path command line arguments. 

### generateTestData.py

This script is a wrapper for calling mismatch_data_gen.py. The script accepts various command line 
arguments which will determine the type of dataset generated. The parameters to this function are as follows:


  ```
  --simLength SIMLENGTH
                        the number of batches of data to generate
                        (default=1000)
  --batchSize BATCHSIZE
                        size of the batches that will be generated
                        (default=32)
                        
  --sequenceLength SEQUENCELENGTH
                        number of samples in each AR process that makes up a
                        batch element (default=20)
                        
  --arVar ARVAR         the variance of the AR Coefficients for each AR
                        process (default=0.1)
                        
  --AR_n AR_N           the order of the AR process that will be generated
                        (default=2){as of this comment, features related to
                        this parameter have not been implmented yet, sochange
                        the value of this from the default at your own peril}
  --cuda                use CUDA for data generation (default: False)
  
  --seed SEED           seed for the rng of the data
```

Upon completion of exeuction, the script will dump the relevant data necessary for neural network training and Kalman 
filter operation into a .mat file which will be specified in the exiting statements of the script. 

EX 1: 

> python generateTestData.py --simLength=1 --batchSize=1 --sequenceLength=1000000 --arVar=0 

```
data saved to:  ./data/data8.mat

it took this many seconds to generate the data 155.63629031181335

Namespace(AR_n=2, arVar=0.0, batchSize=1.0, cuda=False, seed=1111, sequenceLength=1000000.0, simLength=1.0)
```

This script will generate one AR-2 process of length 1000000 samples.

EX 2:

>   python generateTestData.py --simLength=100 --batchSize=32 --sequenceLength=20 --arVar=0  

```
data saved to:  ./data/data8.mat

it took this many seconds to generate the data 10.094983339309692

Namespace(AR_n=2, arVar=0.0, batchSize=32.0, cuda=False, seed=1111, sequenceLength=20.0, simLength=100.0)
```

This script will generate 32*100=3200 different segments of different AR process (the AR variance was specified to be 0
in this example, therefore the same AR parameters are used for each segment).

EX 3:

>   python generateTestData.py --simLength=100 --batchSize=32 --sequenceLength=20 --arVar=0.2

```
data saved to:  ./data/data9.mat

it took this many seconds to generate the data 10.71023154258728

Namespace(AR_n=2, arVar=0.2, batchSize=32.0, cuda=False, seed=1111, sequenceLength=20.0, simLength=100.0)

```

This script will generate 32*100=3200 different segments of different AR processes (the AR variance was specified to be 0.2
in this example, therefore the AR parameters that are used for each segment will have a variance of 0.2 around the mean value).

#### Data Format

Inside of the saved .mat file, you will find four tensors:

- allF - The saved tensor of each and every F matrix used in generating the different AR processes. This value is saved
only to get a value for the lower bound of what the Kalman filter can possibly
predict, given knowledge of the actual AR parameters.  

- allTrueStateValues - A 5 dimensional tensor of all the true, AR process state values. The dimensions in order are: <br />
[batchSize, iteration index, imaginary/complex couple, sequence length+1, series length] 

- measuredData - A 4 dimensional tensor of all the measure AR process state values. The dimensions in order are: <br />
[batchSize, imaginary/complex couple, sequence length, series length]

- predAndCurState - A tensor of the true, predicted and current state values (these values are pulled from the 
allTrueStateValues tensor to make it easier to input for the neural network training) The dimensions in order are: <br />
[batchSize, imaginary/complex couple (1, 3) estimation (2, 4) prediction, sequence length, series length]

