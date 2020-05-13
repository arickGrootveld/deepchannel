clear;
clc;
load('./ARMismatchPredictor/data/data1.mat');
[batchSize, ~, seqLength, seriesLength] = size(measuredData);

measuredConvertedData = zeros(batchSize*seriesLength, seqLength);

for i=1:seriesLength
    for m=1:batchSize
        measuredConvertedData(m+((i-1)*batchSize), :) = ... 
            measuredData(m,1,:,i) + measuredData(m,2,:,i)*1j;
    end
end
