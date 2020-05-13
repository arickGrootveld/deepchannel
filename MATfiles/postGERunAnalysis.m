function postGERunAnalysis(logFile)
    % Getting the path to the data file and the instantaneous errors
    % from the log file
    load(logFile, 'testDataFile', 'testInfo')
    
    instSqErr = testInfo{1,1}.instantaneousSquaredErrors;
    
    % Parsing out the path to the data file from the log files path
    splitLogString = strsplit(logFile, '/');
    dataFilePath = strcat(strcat(strjoin( ...
    splitLogString(1:end-2), '/'), '/'), testDataFile);
    
    % Getting the state map of which state the Markov Chain is in
    % from the data file
    load(dataFilePath, 'testDataInfo');
    intermediate = testDataInfo(1);
    dataPath = intermediate{1}.dataFilePath;
    load(dataPath, 'stateMap');
    
    % Where the actual magic happens
    flatSqErr = instSqErr(:);
    flatStMap = squeeze(stateMap(:));
    seqLen = length(flatStMap) - length(flatSqErr);
    linedUpFltStMp = flatStMap(seqLen+1:end);
    
    largeIndexes = find(flatSqErr > 5);
    
    transitionRegions = [linedUpFltStMp(test-2), linedUpFltStMp(test-1), ...
        linedUpFltStMp(test), linedUpFltStMp(test+1)];
    
    transitionRegionErrors = [linedUpFltStMp(test-2), ...
        linedUpFltStMp(test-1), linedUpFltStMp(test), ....
        linedUpFltStMp(test+1)];
    
    
    hold on;
    plot(flatSqErr, 'ob');
    
    plot(linedUpFltStMp, 'r');
    
    disp('done');
end