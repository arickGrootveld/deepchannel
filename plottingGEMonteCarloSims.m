% Adding required files to path
addpath('MATfiles');

%% KF0 Loading
% Loading goodLogs if they exist
goodLogFiles = dir('TCN/goodLogs');
goodLogFiles=goodLogFiles(~ismember({goodLogFiles.name},{'.','..'}));

goodKFInc = 0;
if(~isempty(goodLogFiles))
   % This should only happen if their are good KF log files
   goodKFInc = 1;
   KF0PredErrs = loadFromDirStruct(goodLogFiles, 1);
   
end

%% KF1 Loading
% loading bad logs if they exist
badLogFiles = dir('TCN/badLogs');
badLogFiles = badLogFiles(~ismember({badLogFiles.name}, {'.', '..'}));

badKFInc = 0;
if(~isempty(badLogFiles))
   % This should only happen if there are bad KF log files
   badKFInc = 1;
   KF1PredErrs = loadFromDirStruct(badLogFiles, 1);
end

%% LS & TCN Loading
LSandTCNInc = 0;

% Prefer loading from the goodLogs, but will load from either goodLogs
% or badLogs if only one actually exists
if(badKFInc && goodKFInc)
   LSandTCNInc = 1;
   LSPredErrs = loadFromDirStruct(goodLogFiles, 2);
   TCNPredErrs = loadFromDirStruct(goodLogFiles, 0);
elseif(badKFInc)
    LSandTCNInc = 1;
    LSPredErrs = loadFromDirStruct(badLogFiles, 2);
    TCNPredErrs = loadFromDirStruct(badLogFiles, 0);
elseif(goodKFInc)
   LSandTCNInc = 1;
   LSPredErrs = loadFromDirStruct(goodLogFiles, 2);
   TCNPredErrs = loadFromDirStruct(goodLogFiles, 0);
end

%% Genie KF Loading
genKFLogFiles = dir('logs');
genKFLogFiles=genKFLogFiles(~ismember({genKFLogFiles.name},{'.','..'}));

genKFInc = 0;

if(~isempty(genKFLogFiles))
    % If we find that there are genie KF log files
    genKFInc = 1;
    genKFPredErrs = loadFromDirStruct(genKFLogFiles, 3);
end


%%%%%%%%%%%%%%%%%%%%
%% Plotting Stuff %%

disp('here we go plotting again');
if(LSandTCNInc && goodKFInc && badKFInc && genKFInc)
   % plot all the algorithms at once
   fig1 = figure(1);
   hold on;
   plot(genKFPredErrs, '-r*');
   plot(TCNPredErrs, '-bd');
   plot(LSPredErrs, '-g');
   plot(KF0PredErrs, '-c');
   plot(KF1PredErrs, '-m');
   hold off;
   
   xlabel('time');
   ylabel('Mean squared error');
   
   legend('Genie KF', 'TCN', 'LS', 'KF0', 'KF1');
   disp('saving fig to: "plots/everythingRun.fig/.eps');
   savefig(fig1, 'plots/everythingRun1.fig');
   saveas(fig1, 'plots/everythingRun1', 'epsc');
else
    throw('functionality for plotting a subset of the algorithms has not been implemented yet');
end
