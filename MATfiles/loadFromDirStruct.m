% Takes a dir struct as input, along with an index, and loads the data
% straight out of the log file, averaging across all elements
%
% For the dataEnum variable, its values are 
%   0: TCN preds
%   1: KF preds
%   2: LS preds
%   3: Genie KF preds


function loadedData = loadFromDirStruct(dirStruct, dataEnum)
    m = length(dirStruct);
    for i = 1:m
        targFilename = strcat(dirStruct(i).folder, '/', dirStruct(i).name);
        
        if(ismember(dataEnum, [0, 1, 2]))
            testInfo = load(targFilename, 'testInfo');
            testInfo = testInfo.testInfo;
            singleSampleData = loadEnumDataFromDataStruct(testInfo{1,1}, dataEnum);
            
            
        elseif(dataEnum == 3)
            gkfRuns = load(targFilename, 'gkfRuns');
            gkfRuns = gkfRuns.gkfRuns;
            singleSampleData = loadEnumDataFromDataStruct(gkfRuns{1,1}, dataEnum);
        else
            throw('Bad enum passed');
        end
        
        % Averaging the data appropriately
        if(i == 1)
           % If this is the first time through, then preallocate the
           % data variable
           loadedData = singleSampleData/m;
        else
            % Otherwise we average the loaded data value
            loadedData = loadedData + (singleSampleData/m);
        end
       
    end
end

function retData = loadEnumDataFromDataStruct(data, enum)
    
    switch enum
        
        % Dealing with TCN data
        case 0
            retData = data.instantaneousSquaredErrors;
            retData = reshape(retData, [1, size(retData, 1) * size(retData, 2)]); 
            retData = retData(1, 1:end-1);
            
        % Dealing with KF data
        case 1
            retData = data.KFInstaErrs;
            
        % Dealing with LS data
        case 2
            retData = (data.LSInstaErrs)';
            retData = retData(1, 1:end-1);
            
        % Dealing with Genie Data
        case 3
            retData = data.genKFInstaErrs;
            
        otherwise
            throw('Bad enum value input');
    
    end
            
end
