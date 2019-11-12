function logName = matSave(directory, basename, data)
% directory: the directory that the data will be saved in
% basename: the name of the file that the data will be saved to, before
%           the appended numbers
% data: a struct that contains the data to be saved to this file
if ~exist(directory, 'dir')
    mkdir(directory);
end
fileSpaceFound = false;
logNumber = 0;

while ~fileSpaceFound
   logNumber = logNumber+1;
   logName = strcat(directory, '/', basename, num2str(logNumber), '.mat');
   if ~isfile(logName)
       disp(['data saved to:', ' ', logName]);
       fileSpaceFound =  true;
   end
end
save(logName, '-struct', 'data');
end