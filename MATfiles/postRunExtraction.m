function formattedData = postRunExtraction(logFile)
    load(logFile, 'testInfo')
    [~,numTests] = size(testInfo);
    formattedData = zeros(numTests,10);
    
    
    for i=1:numTests
        intermediate = testInfo(i);
        formattedData(i,1) = intermediate{1}.predictionMSE;
        formattedData(i,2) = intermediate{1}.estimationMSE;
        formattedData(i,3) = intermediate{1}.predictionVariance;
        formattedData(i,4) = intermediate{1}.estimationVariance;
        formattedData(i,5) = intermediate{1}.LS_PredMSE;
        formattedData(i,6) = intermediate{1}.LS_EstMSE;
        formattedData(i,7) = intermediate{1}.KF_PredMSE;
        formattedData(i,8) = intermediate{1}.KF_EstMSE;
        formattedData(i,9) = intermediate{1}.riccatiConvergencePred;
        formattedData(i,10) = intermediate{1}.riccatiConvergenceEst;
        
    end
    
    % Extracting the correct save path from the file path passed
    extractedDataFileName = 'extractedData';
    pathSplit = strsplit(logFile, '/');
    [~, pathLen] = size(pathSplit);
    
    savePath = '';
    for i=1:pathLen-1
        savePath = strcat(savePath, pathSplit{i}, '/');
        
    end
    logNum = erase(pathSplit{pathLen}, 'log');
    if(contains(logNum, '.'))
       logNum = strsplit(logNum, '.');
       logNum = logNum{1};
    end
    
    savePath = strcat(savePath, extractedDataFileName, logNum, '.txt');
    
    % Saving the extracted data to the same directory the log was loaded
    % from
    csvwrite(savePath, formattedData);
    
end
    