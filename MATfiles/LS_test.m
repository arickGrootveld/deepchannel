% IMPORTANT NOTE: Run first time with TRAIN=1, then run again with TRAIN=0.
%   (Before running with TRAIN=0, you might adjust N if needed, or maybe uncomment line 8 or 9 if desired.)

%% tweakable system model parameters
p=0.0005;                  % probability of switching between channel states
chan1=[0.3 0.1]';         % good channel
% chan2=[1.949 -0.95]';      % bad channel
chan2=[1.95 -0.95]';      % bad channel
% chan2=chan1;             % useful during test (not TRAIN) mode to test LS on all good channels
% chan1=chan2;             % useful during test (not TRAIN) mode to test LS on all bad channels
sigma_v2=0.1;             % variance of v (i.e., process noise)
sigma_w2=0.1;             % variance of measurement noise, decreases with time (remaps Kay's time indices)
N=1e7;                    % length of training sequence
TRAIN=0;                  % when set to one, trains LS model to create LS predictor; when set to zero, assumes "a_ls_pred" variable is in workspace and computes performance

% plotting parameters
N_plot=1e4;      % number of points to plot (must be <= N)
scale=700;       % scaling on GE state indicator

%% initializations
x=zeros(2,N);                                                         % generate space for x
z=zeros(1,N);                                                         % generate space for z
GEState=zeros(N,1);                                                   % generate space for GEState
v=[(randn(1,N)+1j*randn(1,N))/sqrt(2); zeros(1,N)]*sqrt(sigma_v2);    % generate process noise
w=(randn(1,N)+1j*randn(1,N))/sqrt(2).*sqrt(sigma_w2);                 % generate measurement noise
x(:,1)=randn(2,1)/sqrt(2)+1j*randn(2,1)/sqrt(2);                      % initialize x
GEState(1)=round(rand);                                               % initialize current channel state (good/bad)

%% main loop to generate channel state process
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


%% compute LS solution (if TRAIN=1), or compute test performance and plot results (if TRAIN=0)
if TRAIN
    % compute LS predictor
    windowSize=14;
    a_ls_pred=pinv(toeplitz(z(windowSize:N-1),z(windowSize:-1:1)))*x(1,windowSize+1:N).';
    % a_ls_pred=real(a_ls_pred);  % optional, but can toss the small imaginary component... might make for minimal improvement [not sure students are doing this?] or at least a computational savings
    
else % TEST
    sq_error=abs(toeplitz(z(windowSize:N-1),z(windowSize:-1:1))*a_ls_pred-x(1,windowSize+1:N).').^2;
    plot(windowSize+1:N_plot,sq_error(1:N_plot-windowSize),'o',1:N_plot,GEState(1:N_plot)*scale,'LineWidth',2)
    legend('squared error','bad channel indicator')
    xlabel('time')
    grid on
    
    % compute MMSE for each channel
    M_pred_ric1=dare([chan1'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);
    M_pred_ric2=dare([chan2'; 1 0]',[1; 0],[sigma_v2 0; 0 0],sigma_w2);
    
    % output LS MSE and MMSE results
    LS_MSE=mean(sq_error)
    MMSE=M_pred_ric1(1,1)/2+M_pred_ric2(1,1)/2  % ignores transitions
end
