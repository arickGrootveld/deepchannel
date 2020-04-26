% Modified by Kirty Vedula - April 2020.
%
% Tracking a Target with Simple Maneuvering demonstration
%
% Simple demonstration for linear IMM with:
%  1. Nearly constant velocity model
%  2. Nearly constant acceleration model
%
% The measurement model is linear and noisy measurements of target's position.

clc
clear

% Dimensionality of the state space
fdims = 6;
hdims = 2;
nmodels = 2;

% Process noise variance
q{1} = 10^(-8);
Qc{1} = diag([q{1} q{1}]);

q{2} = 1;
Qc{2} = diag([q{2} q{2}]);

% Measurement noise variance
r{1} = 0.1;
R{1} = diag([r{1} r{1}]);

r{2} = 0.1;
R{2} = diag([r{2} r{2}]);

% Transition matrix for the continous-time velocity model.
f{1} = [0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];
f{2} = [0 0 1 0 0 0;  0 0 0 1 0 0; 0 0 0 0 1 0;  0 0 0 0 0 1; 0 0 0 0 0 0;  0 0 0 0 0 0];

% Noise effect matrix for the continous-time system.
L{1} = [0 0; 0 0; 1 0; 0 1];
L{2} = [0 0; 0 0; 0 0; 0 0; 1 0; 0 1];

% Measurement models.
H{1} = [1 0 0 0; 0 1 0 0];
H{2} = [1 0 0 0 0 0; 0 1 0 0 0 0];

dt = 1; % Stepsize


% Discretization of the continous-time system.
for i=1:nmodels
    [F{i},Q{i}] = lti_disc(f{i},L{i},Qc{i},dt);
end
% Index
ind{1} = [1 2 3 4]';
ind{2} = [1 2 3 4 5 6]';

% Generate the data.
n = 200;

X_r = zeros(fdims,n);
X_r(:,1) = zeros(size(F{2},1),1);
Q_r = zeros(fdims,n);

Y = zeros(hdims,n);
R_r = zeros(hdims,n);

mstate = zeros(1,n);

w = [0.5 0.5];
alpha = 0.5;
p_ij = [alpha 1-alpha; 1-alpha alpha];

% Forced mode transitions
mstate(1:50) = 1;
mstate(51:70) = 2;
mstate(71:120) = 1;
mstate(121:150) = 2;
mstate(151:200) = 1;

% Get noise model
for i=1:nmodels
    gen_sys_noise{i} = @(u) mvnrnd(zeros(1,size(Q{i},1)),Q{i},1)';
    gen_obs_noise{i} = @(v) mvnrnd(zeros(1,size(R{i},1)),R{i},1)';
end

% Process model
for i = 2:n
    st = mstate(i);
    Q_r(ind{st},i) = gen_sys_noise{st}();
    X_r(ind{st},i) = F{st}*X_r(ind{st},i-1) + Q_r(ind{st},i);
end


% Generate the measurements.
for i = 1:n
    st = mstate(i);
    R_r(:,i) = gen_obs_noise{st}();
    Y(:,i) = H{st}*X_r(ind{st},i) + R_r(:,i);
end

%% Plot Measurement vs True trajectory
h = plot(Y(1,:),Y(2,:),'ko',X_r(1,:),X_r(2,:),'g-');
legend('Measurement','True trajectory');
xlabel('x');
ylabel('y');
set(h,'markersize',2);
set(h,'linewidth',1.5);

[finalStateValues, observedStates] = reformatIMMData(X_r, Y);

save('data/test.mat', 'finalStateValues', 'observedStates');
