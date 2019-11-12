function [reformatedData, dataInfo] = AKtoAGDataConversion(x, z, ls_coeffs, badRicPred, goodRicPred, batchSize)

[windowSize, ~] = size(ls_coeffs);
[~,N] = size(z);

observedStates = zeros(2, windowSize, N - windowSize - 1);

toepObs = transpose(fliplr(toeplitz(z(windowSize:N-1),z(windowSize:-1:1))));
observedStates(1,:,:) = real(toepObs(:,1:end-1));
observedStates(2,:,:) = imag(toepObs(:,1:end-1));

x_flat = x(1,:);
systemStates = zeros(2, windowSize+1, N-windowSize-1);
toepSysSts = transpose(fliplr(toeplitz(x_flat(windowSize+1:N-1),x_flat(windowSize+1:-1:1))));
systemStates(1,:,:) = real(toepSysSts);
systemStates(2,:,:) = imag(toepSysSts);

finalStateValues = zeros(4, N-windowSize-1);
% Current State Values
finalStateValues(1,:) = systemStates(1,end,:);
finalStateValues(3,:) = systemStates(2,end,:);
% Next State Values
finalStateValues(2,:) = systemStates(1,end,:);
finalStateValues(4,:) = systemStates(2,end,:);

riccatiConvergences = [goodRicPred(1,1), 0;badRicPred(1,1),0];

reformatedDataSaveFormat.riccatiConvergences = riccatiConvergences;
reformatedDataSaveFormat.finalStateValues = finalStateValues;
reformatedDataSaveFormat.systemStates = systemStates;
reformatedDataSaveFormat.observedStates = observedStates;
logname = matSave('data', 'GEDataAK', reformatedDataSaveFormat);

dataInfo.dataFilePath = logname;
% This isn't the correct riccatiConvergence for all possible variations 
% of the code that will be run (i.e. dont trust this) but should work for
% a prototype
dataInfo.riccatiConvergencePred = riccatiConvergences(1,1);
reformatedData = {systemStates, observedStates, finalStateValues};
end