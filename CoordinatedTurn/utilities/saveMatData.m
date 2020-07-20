function filename = saveMatData(data, filePath, fileSubName)
    % Creating the target directory if it doesnt already exist
    if ~exist(filePath, 'dir')
        mkdir(filePath);
    end
    
    d = dir ('data');
    d([d.isdir])= []; %Remove all directories.
    names = setdiff({d.name},{'.','..'});
    
    if(filePath(end) ~= '/')
        filePath = strcat(filePath, '/');
    end
    
    if(isempty(names))
        filename = strcat(filePath, fileSubName, num2str(0), '.mat');
        
        
    else
        m = 0;
        nameFound = 0;
        while nameFound == 0
           filename = strcat(fileSubName, num2str(m), '.mat');
           if(~contains(names, filename))
              nameFound = 1;
           else
              m = m + 1;
           end
        end
        filename = strcat(filePath, filename);
        disp('ha, gottem');
    end
    
    save(filename, 'data');
    disp(strcat('data saved to: ', filename));
end