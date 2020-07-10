% Reformat the data from Kirty's Maneurvering Targets code to TCN format
function [finStates, obsStates] = reformatManTargData(X_r, Y, seqLen)
    numSamps = length(X_r);
    numSeqs = numSamps - seqLen;
    
    finStates = zeros(4, numSeqs);
    obsStates = zeros(2, seqLen, numSeqs);
    
    inter1 = X_r(1:2, :);
    inter2 = Y(1:2, :);
    
    for i = 1:numSeqs
       finStates(1, i) = X_r(1, i + seqLen);
       finStates(3, i) = X_r(2, i + seqLen);
       obsStates(:, :, i) = Y(:,i:i+seqLen-1);
    end

end