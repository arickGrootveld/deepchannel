% IMPORTANT NOTE: Run first time with TRAIN=1, then run again with TRAIN=0.
%   (Before running this script you need to generate a_ls_pred from the 
%    LS_test script)

%% tweakable system model parameters
p=0.0005;                  % probability of switching between channel states
chan1_o=[0.3 0.1]';         % good channel
chan2_o=[1.95 -0.95]';      % bad channel
chan1 = chan1_o;
chan2 = chan2_o;
% chan2=chan1;            % useful during test (not TRAIN) mode to test LS on all good channels
% chan1=chan2;            % useful during test (not TRAIN) mode to test LS on all bad channels
sigma_v2=0.1;             % variance of v (i.e., process noise)
sigma_w2=0.1;             % variance of measurement noise, decreases with time (remaps Kay's time indices)
N=2e6;                    % length of training sequence
TRAIN=0;                  % when set to one, trains LS model to create LS predictor; when set to zero, assumes "a_ls_pred" variable is in workspace and computes performance

% plotting parameters
N_plot=1e4;      % number of points to plot (must be <= N)
scale=700;       % scaling on GE state indicator


batchSize=32;
windowSize=14;
numBatchedSequences = floor((N-windowSize-1)/batchSize);

% Global Initializations
LSandKFTestData = [];
testDataInfo = [];
trueStateTEST = zeros(3,batchSize, 4, numBatchedSequences-1);
measuredStateTEST = zeros(3, batchSize, 2, windowSize, numBatchedSequences-1);


%% Gilbert-Elliot Scenario
% initializations1
x=zeros(2,N);                                                         % generate space for x
z=zeros(1,N);                                                         % generate space for z
GEState=zeros(N,1);                                                   % generate space for GEState
v=[(randn(1,N)+1j*randn(1,N))/sqrt(2); zeros(1,N)]*sqrt(sigma_v2);    % generate process noise
w=(randn(1,N)+1j*randn(1,N))/sqrt(2).*sqrt(sigma_w2);                 % generate measurement noise
x(:,1)=randn(2,1)/sqrt(2)+1j*randn(2,1)/sqrt(2);                      % initialize x
GEState(1)=round(rand);                                               % initialize current channel state (good/bad)

% main loop to generate channel state process1
for n=2:N
    % switch channels if MC changes state
    if rand<p
        GEState(n)=~GEState(n-1); % Gilbert Elliot channel state flips
    else
        GEState(n)=GEState(n-1);
    end
    
    % update channel state vector
    if ~GEState(n)
        F=[chan1'; 1 0];
    else
        F=[chan2'; 1 0];
    end
    x(:,n)=F*x(:,n-1)+v(:,n);
    
    % update observation
    z(n)=x(1,n)+w(n);
    
end

sq_error=abs(toeplitz(z(windowSize:N-1),z(windowSize:-1:1))*a_ls_pred-x(1,windowSize+1:N).').^2;

% compute MMSE for each channel
M_pred_ric1=dare([chan1'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);
M_pred_ric2=dare([chan2'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);

% output LS MSE and MMSE results
disp('Gilbert-Elliot Scenario MSEs');
LS_MSE=mean(sq_error)
MMSE=M_pred_ric1(1,1)/2+M_pred_ric2(1,1)/2  % ignores transitions

[subsetTestData, subsetTestDataInfo] = AKtoAGDataConversion(x, z, a_ls_pred, M_pred_ric2, M_pred_ric1, batchSize);

[trueStateTEST(1,:,:,:), measuredStateTEST(1,:,:,:,:)] = ...
    convertToBatched(subsetTestData{1,3}, subsetTestData{1,2}, batchSize);

testDataInfo = [testDataInfo, subsetTestDataInfo];
LSandKFTestData = [LSandKFTestData, subsetTestData];
%% Only Good States
chan1 = chan1_o;
chan2 = chan1_o;
% initializations1
x=zeros(2,N);                                                         % generate space for x
z=zeros(1,N);                                                         % generate space for z
GEState=zeros(N,1);                                                   % generate space for GEState
v=[(randn(1,N)+1j*randn(1,N))/sqrt(2); zeros(1,N)]*sqrt(sigma_v2);    % generate process noise
w=(randn(1,N)+1j*randn(1,N))/sqrt(2).*sqrt(sigma_w2);                 % generate measurement noise
x(:,1)=randn(2,1)/sqrt(2)+1j*randn(2,1)/sqrt(2);                      % initialize x
GEState(1)=round(rand);                                               % initialize current channel state (good/bad)

% main loop to generate channel state process2
for n=2:N
    % switch channels if MC changes state
    if rand<p
        GEState(n)=~GEState(n-1); % Gilbert Elliot channel state flips
    else
        GEState(n)=GEState(n-1);
    end
    
    % update channel state vector
    if ~GEState(n)
        F=[chan1'; 1 0];
    else
        F=[chan2'; 1 0];
    end
    x(:,n)=F*x(:,n-1)+v(:,n);
    
    % update observation
    z(n)=x(1,n)+w(n);
    
end

sq_error=abs(toeplitz(z(windowSize:N-1),z(windowSize:-1:1))*a_ls_pred-x(1,windowSize+1:N).').^2;

% compute MMSE for each channel
M_pred_ric1=dare([chan1'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);
M_pred_ric2=dare([chan2'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);

% output LS MSE and MMSE results
disp('Only Good State MSEs');
LS_MSE=mean(sq_error)
MMSE=M_pred_ric1(1,1)/2+M_pred_ric2(1,1)/2  % ignores transitions
[subsetTestData, subsetTestDataInfo] = AKtoAGDataConversion(x, z, a_ls_pred, M_pred_ric2, M_pred_ric1, batchSize);

% [batchedFinalStateValues, batchedObservedStates] = ...
%     convertToBatched(subsetTestData{1,3}, subsetTestData{1,2}, batchSize);

[trueStateTEST(2,:,:,:), measuredStateTEST(2,:,:,:,:)] = ...
    convertToBatched(subsetTestData{1,3}, subsetTestData{1,2}, batchSize);

testDataInfo = [testDataInfo, subsetTestDataInfo];
LSandKFTestData = [LSandKFTestData, subsetTestData];
%% Only Bad States
chan1 = chan2_o;
chan2 = chan2_o;
% initializations1
x=zeros(2,N);                                                         % generate space for x
z=zeros(1,N);                                                         % generate space for z
GEState=zeros(N,1);                                                   % generate space for GEState
v=[(randn(1,N)+1j*randn(1,N))/sqrt(2); zeros(1,N)]*sqrt(sigma_v2);    % generate process noise
w=(randn(1,N)+1j*randn(1,N))/sqrt(2).*sqrt(sigma_w2);                 % generate measurement noise
x(:,1)=randn(2,1)/sqrt(2)+1j*randn(2,1)/sqrt(2);                      % initialize x
GEState(1)=round(rand);                                               % initialize current channel state (good/bad)

% main loop to generate channel state process3
for n=2:N
    % switch channels if MC changes state
    if rand<p
        GEState(n)=~GEState(n-1); % Gilbert Elliot channel state flips
    else
        GEState(n)=GEState(n-1);
    end
    
    % update channel state vector
    if ~GEState(n)
        F=[chan1'; 1 0];
    else
        F=[chan2'; 1 0];
    end
    x(:,n)=F*x(:,n-1)+v(:,n);
    
    % update observation
    z(n)=x(1,n)+w(n);
    
end

sq_error=abs(toeplitz(z(windowSize:N-1),z(windowSize:-1:1))*a_ls_pred-x(1,windowSize+1:N).').^2;

% compute MMSE for each channel
M_pred_ric1=dare([chan1'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);
M_pred_ric2=dare([chan2'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);

% output LS MSE and MMSE results
disp('Only Bad State MSEs');
LS_MSE=mean(sq_error)
MMSE=M_pred_ric1(1,1)/2+M_pred_ric2(1,1)/2  % ignores transitions
[subsetTestData, subsetTestDataInfo] = AKtoAGDataConversion(x, z, a_ls_pred, M_pred_ric2, M_pred_ric1, batchSize);

[trueStateTEST(3,:,:,:), measuredStateTEST(3,:,:,:,:)] = ...
    convertToBatched(subsetTestData{1,3}, subsetTestData{1,2}, batchSize);

testDataInfo = [testDataInfo, subsetTestDataInfo];
LSandKFTestData = [LSandKFTestData, subsetTestData];

%% Saving data in proper form
testDataToBeSaved.trueStateTEST = trueStateTEST;
testDataToBeSaved.measuredStateTEST = measuredStateTEST;
testDataToBeSaved.testDataInfo = testDataInfo;
testDataToBeSaved.LSandKFTestData = LSandKFTestData;
testFile = matSave('data', 'GETestDataAK', testDataToBeSaved);

% Seeing errors when loading the data from matlab to the python
% script because the stucts that MATLAB generates are being removed
% when opened with the python code. Good luck


