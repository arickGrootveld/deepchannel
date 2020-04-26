function manTargTrainSave(finalStateValues, observedStates)
    
    targDirName = 'data/';
    dataFiles = dir(targDirName);
    dataFiles=dataFiles(~ismember({dataFiles.name},{'.','..'}));
    
    filenameList = {};
    % Construct cell array of file names inside the data directory
    for i = 1:length(dataFiles)
       filenameList = [filenameList, dataFiles(i).name'];
    end

    targStartName = 'manTargData';
    targEndName = '.mat';
    
    m = 0;
    filenameFound = 0;
    
    % While loop to find an unused file name for the data to be saved
    while filenameFound == 0
        targFilePath = strcat(targStartName, num2str(m), targEndName);
        
        
        if(find(ismember(filenameList, targFilePath)))
            m = m + 1;
        else
            filenameFound = 1;
            finalFilePath = strcat(targDirName, targFilePath);
        end
        
    end
    
    disp(['Saving data to:', ' ', finalFilePath]);
    save(finalFilePath, 'finalStateValues', 'observedStates');
    
end