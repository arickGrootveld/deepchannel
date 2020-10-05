%% Version history:
% v1.0, 9/5/2020, AGKlein, Initial version
% v1.1, 9/6/2020, AGKlein, Fixed unwanted zeros in creation of XX, added
%                          flag to intentionally add zeros to XX for testing
%                          transient MSE performance, added plots of genie KF MSE

%% Description:
% This script is designed to generate flight trajectories using the maneuvering targets model discussed
% in Section 11.7 of Bar-Shalom's "Estimation with Applications to Tracking and Navigation".  Where
% possible, the same notation was used.  The script also includes a genie Kalman filter, an IMM, and
% generates formatted data for use in training and testing a neural network (NN).


%% tweakable system parameters
numSims=10;                 % number of realizations 
N=1e2;                      % length of each realization 
T=1;                        % radar sampling interval (seconds)
P=[0.99 0.01; 0.01 0.99];   % mode transition probabilities into/out of each mode, rows must sum to 1 (if T changes, avg sojourn time in each mode changes)
velocity_init_mean = 120;   % mean of initial velocity
velocity_init_var = 30^2;   % variance of initial velocity
Om_init_mean = 4*pi/180;    % mean of initial turn rate each time system enters CT state (rad/sec)
Om_init_var = (pi/180)^2;   % variance of initial turn rate
Q = 1*eye(2);               % covariance of process noise (position and velocity, v1) [could prob crank this up?]
sigma_v2_2 = (.2*pi/180)^2; % (co)variance of process noise (turn rate, v2)
R = (0.2*velocity_init_mean)^2*eye(2);  % covariance of measurement noise [maybe makes sense to increase w/velocity?]
seed = 102;                 % Setting the seed of the rng state

%% tweakable data generation parameters
seqLength = 20;             % length of each sequence input to NN
outfile = 'data/output.mat';     % output filename for storing XX and YY variables for NN
SHUFFLE = false;             % shuffles data (across realizations) before saving
INCLUDE_ZEROS = false;      % includes initial transient by pre-pending seqLength-1 zeros (useful for showing MSE vs. time on a fully trained NN, generally used with SHUFFLE = false)
rng(seed);

%% non-tweakable parameters and intermediate variables
G=[T^2/2 0; T 0; 0 T^2/2; 0 T];  % matrix for system model (state eq)
H=[1 0 0 0; 0 0 1 0];            % matrix for system model (observation eq)
numModes = size(P,1);            % number of modes
if INCLUDE_ZEROS
    N2=N+2;                    % need extra samples since time zero isn't predicted, nor is last observation used
else
    N2=N+seqLength+1;          % need extra samples since we're skipping initial transient for NN
end


%% allocate space and initialize
x=zeros(4,N2);                    % allocate space for x (state vector)
Om=zeros(N2,1);                   % allocate space for Om (turn rate)
z=zeros(2,N2);                    % allocate space for z (observation)
x_hat=zeros(4,N2);                % allocate space for x_hat (estimate), and implictly initialize [zero mean assumption]
x_hat_pred=zeros(4,N2);           % allocate space for x_hat_pred (prediction)
K=zeros(4,2,N2);                  % allocate space for K (genie Kalman gain)
M=zeros(4,4,N2);                  % allocate space for M (minimum mean squared error), and initialize
temp=(velocity_init_mean^2+velocity_init_var)/2;  % temp var used in initializing M
M(:,:,1)=diag([0 temp 0 temp]);  % initialize M
M_pred=zeros(4,4,N2);             % allocate space for M_pred (minimum predicted mean squared error)
sampleM=zeros(4,4,N2);            % allocate space for sampleM (sample minimum mean squared error)
sampleM_pred=zeros(4,4,N2);       % allocate space for sampleM_pred (sample minimum predicted mean squared error)
XX=zeros(N*numSims, seqLength, 2);  % allocate space for saved data for NN (inputs)
YY=zeros(N*numSims, 2);             % allocate space for saved data for NN (outputs)

%% generate all random variables
v1_all=zeros(2,N2,numSims);
w_all=zeros(2,N2,numSims);
for i=1:numSims
    v1_all(:,:,i)=sqrtm(Q)*randn(2,N2);        % generate all process noise for position/velocity
    w_all(:,:,i)=sqrtm(R)*randn(2,N2);         % generate all measurement noise
end
v2_all=sqrt(sigma_v2_2)*randn(N2,numSims);  % generate all process noise for turn rate
magvel=randn(1,numSims)*sqrt(velocity_init_var)+velocity_init_mean;  % mag of init velocity is Gaussian
direction=rand(1,numSims)*2*pi;            % direction of init velocity is uniform on (0,2pi)
x_init=[zeros(1,numSims); magvel.*cos(direction); zeros(1,numSims); magvel.*sin(direction)]; % random initial state vector (assumes origin is starting position w.l.o.g.)
mode_all = zeros(N2,numSims);               % generate Markov chains to indicate current mode
mode_all(1,:) = randi(numModes, 1, numSims);
CP = cumsum(P,2);
for n=2:N2
    [~,mode_all(n,:)] = max(rand(numSims,1)<CP(mode_all(n-1,:),:),[],2);
end
mode_all=mode_all-1; % subtract 1 to index modes by 0, 1, ...
% To only run in CV mode, uncomment this --
%mode_all = zeros(size(mode_all));
% To only run in CT mode, uncomment this --
%mode_all = ones(size(mode_all));
maxTurns=max(floor(sum((diff(mode_all))~=0)/2)+1);
Om_init=(randn(maxTurns,numSims)*sqrt(Om_init_var)+Om_init_mean).*sign(randn(maxTurns,numSims));  % random initial turn rate at start of each turn, folded normal distribution to account for L or R turns


%% main loop
for k=1:numSims
    
    %% extract random variables for kth realization
    v1=v1_all(:,:,k);
    v2=v2_all(:,k);
    w=w_all(:,:,k);
    x(:,1)=x_init(:,k);
    mode=mode_all(:,k);
    turnCtr=0;  % initialize turn counter
    
    %% main loop
    for n=2:N2
        %% system update equations
        % update turn rate state Om
        if mode(n) && ((mode(n) ~= mode(n-1)) || n==2) % just entered CT mode
            turnCtr=turnCtr+1;
            Om(n)=Om_init(turnCtr,k);
        elseif mode(n) % CT mode (but not first time entering this mode)
            Om(n)=Om(n-1)+T*v2(n-1);
        else % CV mode
            Om(n)=0;
        end
        
        % update all other states and observation
        F = getF(Om(n),T);
        x(:,n)=F*x(:,n-1)+G*v1(:,n-1);
        z(:,n)=H*x(:,n)+w(:,n);
        
        % Genie Kalman filter updates (implictly has knowledge of turn rate)
        x_hat_pred(:,n)=F*x_hat(:,n-1);
        M_pred(:,:,n)=F*M(:,:,n-1)*F'+G*Q*G';
        K(:,:,n)=M_pred(:,:,n)*H'/(H*M_pred(:,:,n)*H'+R);
        x_hat(:,n)=x_hat_pred(:,n)+K(:,:,n)*(z(:,n)-H*x_hat_pred(:,n));
        M(:,:,n)=(eye(4)-K(:,:,n)*H)*M_pred(:,:,n);
        
        % compute sample mean square error (i.e., actual, not theoretical)
        e=x_hat(:,n)-x(:,n);               % error between estimate and true state
        e_pred=x_hat_pred(:,n)-x(:,n);     % error between prediction and true state
        sampleM(:,:,n)=sampleM(:,:,n)+e*e'/numSims;  % sample MSE
        sampleM_pred(:,:,n)=sampleM_pred(:,:,n)+e_pred*e_pred'/numSims;  % sample prediction MSE
        
    end
    
    % store data for NN training / test
    if INCLUDE_ZEROS
        X=cat(3,toeplitz([z(1,2) zeros(1,seqLength-1)]',z(1,2:end-1))', toeplitz([z(2,2) zeros(1,seqLength-1)]',z(2,2:end-1))');
        Y=[x(1,3:end)' x(3,3:end)'];
    else
        X=cat(3,toeplitz(z(1,seqLength+1:-1:2),z(1,seqLength+1:end-1))', toeplitz(z(2,seqLength+1:-1:2),z(2,seqLength+1:end-1))');
        Y=[x(1,seqLength+2:end)' x(3,seqLength+2:end)'];
    end
    offset=X(:,end,:);
    XX((k-1)*N+1:k*N,:,:)=X-offset;
    YY((k-1)*N+1:k*N,:,:)=Y-squeeze(offset);
    
end

%% save data to file for NN
if SHUFFLE % useful for training, not so useful for test and debug
    idx=randperm(N*numSims);
    XX=XX(idx,:,:);
    YY=YY(idx,:);
end
save(outfile,'XX','YY')

%% report performance of algorithms
disp(['Averaged over ' num2str(numSims) ' realizations with ' num2str(N2-seqLength-1) ' samples each, the mean'])
disp('squared prediction error (x and y coords combined) is as follows:')
disp(' ')
disp(['genie KF: ' num2str(mean(sampleM_pred(1,1,seqLength+2:end))+mean(sampleM_pred(3,3,seqLength+2:end)))])
disp(['LS: ' 'TBD'])
disp(['IMM: ' 'TBD'])

%% plot trajectory of last realization
figure(1)
set(0, 'DefaultLineLineWidth', 2);
plot(x(1,mode==0),x(3,mode==0),'ko')
hold on
plot(x(1,mode==1),x(3,mode==1),'ks','Linewidth',4)
plot(z(1,:),z(2,:),'co')
plot(x_hat_pred(1,:),x_hat_pred(3,:),'mo')
plot(x(1,:),x(3,:),'k-','Linewidth',1)
plot(0,0,'g*','Linewidth',8)
hold off
legend('true state (CV mode)','true state (CT mode)','noisy observation','genie KF prediction')
axis equal
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title(['target trajectory (duration: ' num2str(round(N2*T/60*10)/10) ' minutes)'])

%% plot turn rate and velocity of last realization
figure(2)
subplot(211)
t=(0:N2-1)*T/60;
plot(t,sqrt(x(2,:).^2+x(4,:).^2))
grid on
xlabel('time (minutes)')
ylabel('speed (m/s)')
title('airspeed (top) and turn rate (bottom)')
subplot(212)
plot(t,Om)
grid on
xlabel('time (minutes)')
ylabel('turn rate (radians/s)')

%% plot MSE performance
figure(3)
plot(0:N2-2,squeeze(M_pred(1,1,2:end))+squeeze(M_pred(3,3,2:end)),0:N2-2,squeeze(sampleM_pred(1,1,2:end))+squeeze(sampleM_pred(3,3,2:end)))
xlabel('discrete time sample index')
ylabel('mean squared error')
grid on
legend('theoretical genie KF MSE', 'experimental genie KF MSE')
title(['sum of MSE on x and y coordinates over all ' num2str(numSims) ' runs'])


%% TODO:
% + Add IMM
% + Given an output file of predicted YY's from NN in .mat format, compute MSE


%% Define F matrix which is a function of Om (turn rate, possibly time-varying) and T (sample interval, a constant)
function F = getF(Om, T)
if Om
    F = [1 sin(Om*T)/Om 0 -(1-cos(Om*T))/Om; 0 cos(Om*T) 0 -sin(Om*T); 0 (1-cos(Om*T))/Om 1 sin(Om*T)/Om; 0 sin(Om*T) 0 cos(Om*T)];
else % Om=0, not turning
    F = [1 T 0 0; 0 1 0 0; 0 0 1 T; 0 0 0 1];
end
end
