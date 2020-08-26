%% Dimensionality of the state space
fdims = 5; %[x1 x2 v1 v2 w]
hdims = 2;
nmodels = 2;

n = 30000;
seed = 6333;
rng(seed);

tic;
%% Stepsize
dt = 0.1;

%% Transition matrix for the continous-time velocity model.
f{1} = [0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0];
%% Noise effect matrix for the continous-time system.
L{1} = [0 0;
        0 0;
        1 0;
        0 1];
L{2} = [0 0 0 0 1]'; % Noise at turn rate only

%% Process noise variance
q{1} = 0.05;
Qc{1} = diag([q{1} q{1}]);
Qc{2} = 0.15;
Q{2} = L{2}*Qc{2}*L{2}'*dt;

%% Measurement models.
H{1} = [1 0 0 0;
        0 1 0 0];

H{2} = [1 0 0 0 0 ;
        0 1 0 0 0 ];
%% Variance in the measurements.
r{1} = 0.05;
R{1} = diag([r{1} r{1}]);
r{2} = 0.05;
R{2} = diag([r{2} r{2}]);    

%% Discretization of the continous-time system.
[F{1},Q{1}] = lti_disc(f{1},L{1},Qc{1},dt);
F{2} = @c_turn;

%% Index
ind{1} = (1:4)';
ind{2} = (1:5)';

%% Generate the data.
X_r = zeros(fdims,n);
Q_r = zeros(fdims,n);
Y = zeros(hdims,n);
R_r = zeros(hdims,n);
mstate = zeros(1,n);

w1 =  [0.5 0.5];
w2 =  [0.5 0.5];
p_ij = [0.99 0.01; 0.01 0.99];  

%% Probabilistic Mode Transitions
% Randomly choosing a starting mode, each having same probability
mstate(1, 1) = (rand(1,1) < 0.5) + 1;

% Implementation of Markov chain for mode transitions
for i = 2:n
    if(mstate(1,i-1) == 1)
        mstate(1,i) = (rand(1,1) > p_ij(1,1)) + 1;
    else
        mstate(1, i) = (rand(1,1) > p_ij(2,1)) + 1;
    end
end


%% Arick Added code
transitionMatrices = {};
transitionMatrices{1} = f{1};
%
%% Get noise model

for i=1:nmodels
gen_sys_noise{i} = @(u) mvnrnd(zeros(1,size(Q{i},1)),Q{i},1)';   
gen_obs_noise{i} = @(v) mvnrnd(zeros(1,size(R{i},1)),R{i},1)'; 
end

%% Process model
for i = 2:n
   st = mstate(i);
   Q_r(ind{st},i) = gen_sys_noise{st}();
   if isa(F{st},'function_handle')
       [intermediate, transitionMatrices{i}] = F{st}(X_r(ind{st},i-1),dt);
       X_r(ind{st},i) = intermediate + Q_r(ind{st},i);
   else
       X_r(ind{st},i) = F{st}*X_r(ind{st},i-1) + Q_r(ind{st},i);
       transitionMatrices{i} = F{st};
   end

end
%% Generate the measurements.
for i = 1:n
    st = mstate(i);
    R_r(:,i) = gen_obs_noise{st}();
     if isa(H{st},'function_handle')
        Y(:,i) = H{st}(X_r(ind{st},i)) + R_r(:,i);
     else
         Y(:,i) = H{st}*X_r(ind{st},i) + R_r(:,i);
     end
end

toc;
%% Plot Measurement vs True trajectory
% figure(1)
% h = plot(Y(1,:),Y(2,:),'ko', X_r(1,:),X_r(2,:),'g-');
% legend('Measurement', 'True trajectory');
% xlabel('x');
% ylabel('y');
% set(h,'markersize',2,'linewidth',1.5);

%% Initial Values 


% IMM
% Model-conditioned estimates of IMM EKF
MM1_i = cell(2,n);
PP1_i = cell(2,n);
x_ip1{1} = zeros(size(F{1},1),1);
P_ip1{1} = 0.1 * eye(size(F{1},1));
x_ip1{2} = zeros(fdims,1);
P_ip1{2} = 0.1 * eye(fdims);

% Estimates of Genie Kalman Filter
GKF_M = zeros(size(F{1},1),1);
GKF_P = 0.1 * eye(size(F{1},1));

GKF_MM = zeros(size(F{1},1),  n);
GKF_PP = zeros(size(F{1},1), size(F{1},1), n);

GKF_Q{1} = Q{1};
inter1 = Q{2};
GKF_Q{2} = inter1(1:4, 1:4);

%
%% Overall estimates of IMM filter
%%IMMEKF
MM1 = zeros(fdims,  n);
PP1 = zeros(fdims, fdims, n);

%% IMM Model probabilities 
MU1 = zeros(2,n); %IMMEKF

% Clearing last warn to make it easier to detect a warning
%% Filtering steps. %%
for i = 1:n
    
    %IMMEKF
    [x_p1,P_p1,c_j1,predVal] = eimm_predict(x_ip1,P_ip1,w1,p_ij,ind,fdims,F,Q,dt);
    [x_ip1,P_ip1,w1,m1,P1] = eimm_update(x_p1,P_p1,c_j1,ind,fdims,Y(:,i),H,R);
    MM1(:,i)   = predVal;
    PP1(:,:,i) = P1;
    MU1(:,i)   = w1';
    MM1_i(:,i) = x_ip1';
    PP1_i(:,i) = P_ip1';
    
    %Genie KF
    st = mstate(i);
    %% P and m are not right, i.e. first and second variables
    [GKF_M, GKF_P] = kf_predict(GKF_M, GKF_P, transitionMatrices{i}, GKF_Q{st});
    GKF_MM(:,i) = GKF_M;
    GKF_PP(:,:, i) = GKF_P;
    [GKF_M, GKF_P] = kf_update(GKF_M, GKF_P, Y(:,i), H{1}, R{st});
    


end

% Calculating the MSE of the IMM's predictions to be able to 
% compare with TCN 

IMM_MSE = (sum(((X_r(1,11:end)-MM1(1,11:end)).^2 + (X_r(2,11:end) - MM1(2,11:end)).^2)))/length(MM1 - 10);

disp(strcat("IMM Prediction MSE: ", num2str(IMM_MSE)));


GKF_MSE = (sum(((X_r(1,11:end)-GKF_MM(1,11:end)).^2 + (X_r(2,11:end) - GKF_MM(2,11:end)).^2)))/length(GKF_MM - 10);

disp(strcat("Genie Kalman Filter Prediction MSE: ", num2str(GKF_MSE)));


% Plotting squared errors of the IMM and GKF
% figure(7);
% imm_SqErrs = (((X_r(1,11:end)-MM1(1,11:end)).^2 + (X_r(2,11:end) - MM1(2,11:end)).^2));
% gkf_SqErrs = (((X_r(1,11:end)-GKF_MM(1,11:end)).^2 + (X_r(2,11:end) - GKF_MM(2,11:end)).^2));
% hold on
% plot(imm_SqErrs, '-r');
% plot(gkf_SqErrs, '-b');
% hold off;
%% Saving the data

sTrueStates = X_r(1:2,:);
% % Checking to see if the memory requirements are too large to
% % Save them each to their own individual files or if we need
% % to break them up amongst multiple files
% mem1 = whos('sTrueStates');
% mem2 = whos('Y');
% 
% % TODO: Implement the breaking up across multiple files code
% 
save('data/trueStates.mat', 'sTrueStates');
save('data/obsStates.mat', 'Y');
save('data/altAlgPreds.mat', 'GKF_MM', 'MM1');

saveData.riccatiConvergences = [0, 1; 1, 0];
saveData.seed = seed;
% 
saveData.trueStateFiles = ['data/trueStates.mat'];
saveData.obsStateFiles = ['data/obsStates.mat'];
saveData.altAlgFiles = ['data/altAlgPreds'];
save('data/matData.mat', 'saveData')
toc
