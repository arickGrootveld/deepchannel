load('data/GETestData27.mat');
testNum = 1;

observedStates = LSandKFTestData{1,testNum}(2);observedStates = observedStates{1,1};
systemStates = LSandKFTestData{1,testNum}(1);systemStates = systemStates{1,1};

% Converting the data to a format we are familiar with
obsShape = size(observedStates);
sysShape = size(systemStates);

X_r = zeros(4, sysShape(2) + sysShape(3) - 1);
Y = zeros(2, obsShape(2) + obsShape(3));

for m = 1:obsShape(3) - 1
   Y(:, m) = observedStates(:, 1, m+1);
   X_r(1:2,m) = systemStates(:, 1, m+1);
   X_r(3:4,m) = systemStates(:, 1, m);
end

for u = 1:obsShape(2)
   Y(:, obsShape(3) + u - 1) = observedStates(:, u, obsShape(3) - 1);
   X_r(1:2, obsShape(3) + u - 1) = systemStates(:, u, obsShape(3) - 1);
   X_r(3:4, obsShape(3) + u - 1) = systemStates(:, u, obsShape(3) - 2);
end

Y(:, end) = observedStates(:, obsShape(2), obsShape(3));
X_r(1:2, obsShape(3) + obsShape(2)) = systemStates(:, obsShape(2), obsShape(3));
X_r(3:4, obsShape(3) + obsShape(2)) = systemStates(:, obsShape(2), obsShape(3) - 1);


%%% TODO: There is definitely a better way to do the above, so come back to
%%% TODO: it later and clean this up

%% Dimensionality of the state space
fdims = 4; %[x(k)real x(k)cmplx x(k-1)rea x(k-1)cmplx]
hdims = 2;
nmodels = 2;
n = length(Y);

%% Stepsize
dt = 0.1;

%% Transition matrix for the continous-time velocity model.
f{1} = [1.949, 0, -0.95, 0;
        0, 1.949, 0, -0.95;
        1, 0, 0, 0;
        0, 1, 0, 0];
    
f{2} = [0.3, 0, 0.1, 0;
        0, 0.3, 0, 0.1;
        1, 0, 0, 0;
        0, 1, 0, 0];
    
%% Noise effect matrix for the continous-time system.
L{1} = [1 0; 0 0];
L{2} = [1 0; 0 0];

%% Process noise variance
q{1} = 0.1/2;
Qc{1} = diag([q{1} q{1} 0 0]);
Qc{2} = diag([q{1} q{1} 0 0]);

%% Measurement models.
H{1} = [1 0 0 0; 0 1 0 0];

H{2} = [1 0 0 0; 0 1 0 0];
%% Variance in the measurements.
r{1} = 0.1/2;
R{1} = diag([r{1} r{1}]);
r{2} = 0.1/2;
R{2} = diag([r{2} r{2}]);

%% Discretization of the continous-time system.
% [F{1},Q{1}] = lti_disc(f{1},L{1},Qc{1},dt);
% [F{2},Q{2}] = lti_disc(f{2},L{2},Qc{2},dt);
F{1} = f{1};
F{2} = f{2};
Q{1} = Qc{1};
Q{2} = Qc{2};

%% Index
ind{1} = (1:4)';
ind{2} = (1:4)';

%% Generate the data.
% X_r = zeros(fdims,n);
% Q_r = zeros(fdims,n);
% Y = zeros(hdims,n);
% R_r = zeros(hdims,n);
% mstate = zeros(1,n);
% 
w1 =  [0.95 0.05];
w2 =  [0.95 0.05];
p_ij = [0.9995 0.0005; 0.0005 0.9995];

%% Initial Values 
% KF Model 1
KF_M = zeros(size(F{1},1),1);
KF_P = eye(size(F{1},1));

KF_M_2 = zeros(size(F{1},1),1);
KF_P_2 = eye(size(F{1},1));

% IMMEKF (1)
x_ip1{1} = zeros(size(F{1},1),1);
P_ip1{1} = eye(size(F{1},1));
x_ip1{2} = zeros(fdims,1);
P_ip1{2} = eye(fdims);

%% Space For Estimation

% KF Model 1 Filter
KF_MM = zeros(size(F{1},1),  n);
KF_PP = zeros(size(F{1},1), size(F{1},1), n);

KF_MM_2 = zeros(size(F{2},1),  n);
KF_PP_2 = zeros(size(F{2},1), size(F{2},1), n);

% IMM
% Model-conditioned estimates of IMM EKF
MM1_i = cell(2,n);
PP1_i = cell(2,n);


% Overall estimates of IMM filter
%IMMEKF
MM1 = zeros(fdims,  n);
PP1 = zeros(fdims, fdims, n);


% IMM Model probabilities 
MU1 = zeros(2,n); %IMMEKF

startIndex = 1;
%% Filtering steps. %%
for i = startIndex:n
    %KF model 1
    [KF_M,KF_P] = kf_predict(KF_M,KF_P,F{1},Q{1});
    KF_MM(:,i)   = KF_M;
    KF_PP(:,:,i) = KF_P;
    [KF_M,KF_P] = kf_update(KF_M,KF_P,Y(:,i),H{1},R{1});
    
    
    % KF model 2
    [KF_M_2,KF_P_2] = kf_predict(KF_M_2,KF_P_2,F{2},Q{2});
    KF_MM_2(:,i)   = KF_M_2;
    KF_PP_2(:,:,i) = KF_P_2;
    [KF_M_2,KF_P_2] = kf_update(KF_M_2,KF_P_2,Y(:,i),H{2},R{2});
    
    
    %IMMEKF
    [x_p1,P_p1,c_j1, predVals] = eimm_predict(x_ip1,P_ip1,w1,p_ij,ind,fdims,F,Q,dt);
    [x_ip1,P_ip1,w1,m1,P1] = eimm_update(x_p1,P_p1,c_j1,ind,fdims,Y(:,i),H,R);
    MM1(:,i)   = predVals;
    PP1(:,:,i) = P1;
    MU1(:,i)   = w1';
    MM1_i(:,i) = x_ip1';
    PP1_i(:,i) = P_ip1';
end


% %% Calculate Normalise Root Mean Square Error (NRMSE)
% %KF Model 1
% NRMSE_KF1_1 = sqrt(mean((X_r(1,:)-KF_MM(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
% NRMSE_KF1_2 = sqrt(mean((X_r(2,:)-KF_MM(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
% NRMSE_KF1 = 1/2*(NRMSE_KF1_1 + NRMSE_KF1_2);
% fprintf('NRMSE of KF1             :%5.2f%%\n',NRMSE_KF1);
% 
% NRMSE_KF2_1 = sqrt(mean((X_r(1,:)-KF_MM_2(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
% NRMSE_KF2_2 = sqrt(mean((X_r(2,:)-KF_MM_2(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
% NRMSE_KF2 = 1/2*(NRMSE_KF2_1 + NRMSE_KF2_2);
% fprintf('NRMSE of KF2             :%5.2f%%\n',NRMSE_KF2);
% 
% %IMM EKF
% NRMSE_IMMEKF1 = sqrt(mean((X_r(1,:)-MM1(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
% NRMSE_IMMEKF2 = sqrt(mean((X_r(2,:)-MM1(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
% NRMSE_IMMEKF = 1/2*(NRMSE_IMMEKF1 + NRMSE_IMMEKF2);
% fprintf('NRMSE of IMMEKF          :%5.2f%%\n',NRMSE_IMMEKF);
% 
% %Original
% NRMSE_ORG1 = sqrt(mean((X_r(1,:)-Y(1,:)).^2))/(max(X_r(1,:))-min(X_r(1,:)))*1e2;
% NRMSE_ORG2 = sqrt(mean((X_r(2,:)-Y(2,:)).^2))/(max(X_r(2,:))-min(X_r(2,:)))*1e2;
% NRMSE_ORG = 1/2*(NRMSE_ORG1 + NRMSE_ORG2);
% fprintf('NRMSE of Original        :%5.2f%%\n',NRMSE_ORG);

mseRange = startIndex:length(X_r);

MSE_ORG = sum(((X_r(1,mseRange) - Y(1, mseRange)).^2) + ...
    ((X_r(2,mseRange) - Y(2, mseRange)).^2)) / numel(mseRange)
      
MSE_IMMEKF = sum(((X_r(1,mseRange) - MM1(1, mseRange)).^2) + ...
    ((X_r(2,mseRange) - MM1(2, mseRange)).^2)) / numel(mseRange)
      
MSE_KF1 = sum(((X_r(1,mseRange) - KF_MM(1, mseRange)).^2) + ...
    ((X_r(2,mseRange) - KF_MM(2, mseRange)).^2)) / numel(mseRange)
      
MSE_KF2 = sum(((X_r(1,mseRange) - KF_MM_2(1, mseRange)).^2) + ...
    ((X_r(2,mseRange) - KF_MM_2(2, mseRange)).^2)) / numel(mseRange)

% myKFValues = zeros(2, length(testInfo{1,testNum}.KFPredVals));
% myKFValues(1, :) = real(testInfo{1,testNum}.KFPredVals);
% myKFValues(2, :) = imag(testInfo{1,testNum}.KFPredVals);

         
% close all;
% viewRange = 1:30;
% figure(1);         
% hold on;
% plot(X_r(1,viewRange), X_r(2, viewRange), '-r');
% plot(MM1(1,viewRange), MM1(2,viewRange), '-*b');
% plot(Y(1,viewRange), Y(2,viewRange), 'ok');
% plot(KF_MM(1,viewRange), KF_MM(2,viewRange), '--g');
% plot(KF_MM_2(1,viewRange), KF_MM_2(2,viewRange), '--m');

% legend('true trajectory', 'IMM', 'Meas States', 'KF1', 'KF2');
% 
% hold off;

