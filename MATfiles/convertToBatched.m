function [trueState, measuredState] = convertToBatched(finalSystemData, observedData, batchSize)

[~, seqLength, numSequences] = size(observedData);

seriesLength = floor(numSequences/batchSize);

% Going to seriesLength-1 because of 
trueState = zeros(batchSize, 4, seriesLength-1);
measuredState = zeros(batchSize, 2, seqLength, seriesLength-1);

for i = 1:seriesLength-1
   trueState(:,:,i) = permute(finalSystemData(:,i*batchSize:((i+1)*batchSize)-1), [3,2,1]);
   % supposed to swap axis and transpose the data here, but not sure if
   % that is unnecesary or not. Return to this if MSE is very high

   measuredState(:,:,:,i) = permute(observedData(:,:, ...
   i*batchSize:((1+i)*batchSize)-1), [3,1,2]);
end



end