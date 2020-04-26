function [finalStateValues, observedStates] = reformatIMMData(X_r, Y)
    % Hard coding the sequene length here for now
    sequenceLength = 10;

    % X_r is the true states
    % Y is the observations

    
    % Gonna end up with M usable states and observation sequences
    % Where M = N - sequenceLength
    % and N is the number of data points we have (the length of Y and X_r)
    M = length(Y) - sequenceLength;
    
    observedStates = teoplitizifyData(Y, sequenceLength);
    
    % Grabbing only the usable states
    observedStates = observedStates(:,:,1:M);
    
%     finalStates = X_r(1:2, 
    for p = 1:M
        % Setting current process state values and next process state
        % values to be in the finalStateValues matrix
        finalStateValues(1, p) = X_r(1, p + sequenceLength - 1);
        finalStateValues(2, p) = X_r(1, p + sequenceLength);
        finalStateValues(3, p) = X_r(2, p + sequenceLength - 1);
        finalStateValues(4, p) = X_r(2, p + sequenceLength);
    end


end