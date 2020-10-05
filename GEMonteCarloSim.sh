#!/bin/bash

## Parameters of the simulation
numSamples=10000
numRuns=500

# Selecting the mode of the Monte Carlo Sim
# 0 is the full GE scenario
# 1 is just the good states
# 2 is just the bad states
mode=$1

# Selecting which algorithms to use for the simulation
# TCN and LS will always be included, but you can select a particular
# type of Kalman Filter to show up by setting its variable to 1
goodKF=1
badKF=1
genieKF=1
IMM=0

if [ $mode -eq 0 ]
then
    echo "mode is 0"
elif [ $mode -eq 1 ]
then
    echo "mode is 1"
elif [ $mode -eq 2 ]
then
    echo "mode is 2"
else
    echo "$mode is out of range for mode, defaulting to mode 0"
    mode=0
fi


# Clear our old data before generating new stuff
rm data/*.mat

## Generating the data based on which mode we are in
for ((i=1; i<=$numRuns; i++ ))
do
    if [ $mode -eq 0 ]
    then
        python gilbertElliotDataGen.py --seed=$RANDOM --simu_len=$numSamples --seq_len=10 --testDataGen --initTest --debug
    elif [ $mode -eq 1 ]
    then
        python gilbertElliotDataGen.py --seed=$RANDOM --noMismatchDataGEn --simu_len=$numSamples --seq_len=10 --testDataGen --initTest --ARCoeffs 0.3 0.1
    else
        python gilbertElliotDataGen.py --seed=$RANDOM --noMismatchDataGEn --simu_len=$numSamples --seq_len=10 --testDataGen --initTest --ARCoeffs 1.949 -0.95
    fi
done

## Moving onto testing algorithm performance against the TCN's
cd TCN

# Clear our old logs
rm logs/*.mat
rm goodLogs/*.mat
rm badLogs/*.mat


# Check TCN and LS performance
for ((i=1; i<=$numRuns; i++ ))
do
    if [ $goodKF -eq 1 ]
    then
      python TCNestimator.py --model_path=models/model131.pth --testDataFile=../data/GETestData$i.mat --debug --cuda --initTest --KFCoeffs 0.3 0.1 --cuda_device=0
      mv logs/log*.mat goodLogs/log$i.mat  
    fi
    
    if [ $badKF -eq 1 ]
    then
        python TCNestimator.py --model_path=models/model131.pth --testDataFile=../data/GETestData$i.mat --debug --cuda --initTest --KFCoeffs 1.949 -0.95 --cuda_device=0
       mv logs/log*.mat badLogs/log$i.mat
    fi
    
    # This if statement is in case neither KF was specified
    if [ $badKF -eq 0 ] && [ $goodKF -eq 0 ]
    then
        python TCNestimator.py --model_path=models/model131.pth --testDataFile=../data/GETestData$i.mat --debug --cuda --initTest --KFCoeffs 0.5 0.4 --cuda_device=0
        mv logs/log*.mat neutralLogs/log$i.mat
    fi
done

if [ $genieKF -eq 1 ]
then
    # Move up a directory to use the Genie KF script
    cd ..
    # Clearing out all previous genie KF run logs
    rm logs/genieLog*.mat
    for ((i=1; i<=$numRuns; i++ ))
    do
        python genieFilterRun.py --testFile=data/GETestData$i.mat --initTest
    done
    cd TCN
fi

if [ $IMM -eq 1 ]
then
    cd ..
    for ((i=1; i<=$numRuns; i++ ))
    do
        matlab -batch "geMonteCarloSim"
    done
fi 



echo "fin"
