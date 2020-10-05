%% Version history:
% v1.0, 9/29/2020, AGKlein, Initial version

%% Description:
% This script plots results of the genie KF, IMM, and TCN for the 
% manuevering targets problem.  Generally, the following sequence
% of scripts will be run:
%  1. maneuvering.m (creates trajectory data)
%  2. imm.m (loads trajectory data, saves result to file)
%  3. TCN code in Python (loads trajectory data, saves result to file)
%  4. mtplot.m (this script, which loads three files from the above steps)

%% tweakable system parameters
infile1 = 'test1.mat';      % file of input data (target trajectories, system parameters)
infile2 = 'test1_imm.mat';  % file of IMM results
infile3 = 'test1_tcn.mat';  % file of TCN results... should contain a single matrix called YY
                            % with length equal to N*numSims -by- 2

%% load data files, rename some stuff
load(infile1)
YYtrue=YY;
x=x_imm;
z=z_imm;
load(infile2)
load(infile3,'YY')
if SHUFFLE
    error('This script not meant for use with SHUFFLE mode.')
end

%% determine some intermediate params
[~,N2,numSims]=size(x_imm);
seqLength=size(XX,2);
if INCLUDE_ZEROS
    N=N2-2;
else
    N=N2-seqLength-1;
end

%% process TCN data (reshape, add bias which was subtracted out, compute MSE)
YY=reshape(YY',2,N,numSims);  % reshape into realizations
if ~INCLUDE_ZEROS
    YY=cat(2,NaN*ones(2,seqLength+1,numSims),YY+z_imm(:,2:N+1,:));  % add bias, and prepend with NaNs where TCN didn't estimate first seqLength time steps
else
    YY=cat(2,NaN*ones(2,2,numSims),YY+cat(2,zeros(2,seqLength-1,numSims),z_imm(:,2:N-seqLength+2,:)));  % prepend with NaNs where TCN didn't estimate first seqLength time steps
end
TCN_MSE = mean(sum((YY-x_imm).^2),3)';             % compute MSE of bias-adjusted NN predicted outputs with true states
x_hat_pred_tcn=YY(:,:,end);                        % grab last realization

%% report performance of algorithms
disp(['Averaged over ' num2str(numSims) ' realizations with ' num2str(N2-seqLength-1) ' samples each, the mean'])
disp('squared prediction error (x and y coords combined) is as follows:')
disp(' ')
disp(['genie KF: ' num2str(mean(GKF_MSE(seqLength+2:end)))])
disp(['     IMM: ' num2str(mean(IMM_MSE(seqLength+2:end)))])
disp(['     TCN: ' num2str(mean(TCN_MSE(seqLength+2:end)))])


%% plot trajectory of last realization
figure(1)
set(0, 'DefaultLineLineWidth', 3);
plot(x(1,mode==0,end),x(2,mode==0,end),'ko')
hold on
plot(x(1,mode==1,end),x(2,mode==1,end),'ks','Linewidth',4)
plot(z(1,:,end),z(2,:,end),'co')
plot(x_hat_pred_GKF(1,:),x_hat_pred_GKF(3,:),'mo')
plot(x_hat_pred_imm(1,:),x_hat_pred_imm(3,:),'go')
plot(x_hat_pred_tcn(1,:),x_hat_pred_tcn(2,:),'bx')
plot(x(1,:,end),x(2,:,end),'k-','Linewidth',1)
plot(0,0,'g*','Linewidth',8)
hold off
legend('true state (CV mode)','true state (CT mode)','noisy observation','genie KF prediction','IMM','TCN')
axis equal
grid on
xlabel('x (meters)')
ylabel('y (meters)')
title(['target trajectory (duration: ' num2str(round(N2*T/60*10)/10) ' minutes)'])

%% plot IMM verification params
figure(2)
subplot(211)
t=(0:N2-1)*T/60;
plot(t,mu(2,:))
grid on
xlabel('time (minutes)')
ylabel('probability')
title('IMM estimated CT mode probability (top) and estimated turn rate (bottom)')
subplot(212)
plot(t,Om,t,Om_imm,t,Om_ct)
grid on
xlabel('time (minutes)')
ylabel('turn rate (radians/s)')
legend('true','IMM mixed estimator','IMM CT estimator')

%% plot MSE performance
figure(3)
plot(0:N2-2,GKF_MSE(2:end),0:N2-2,IMM_MSE(2:end),0:N2-2,TCN_MSE(2:end))
xlabel('discrete time sample index')
ylabel('mean squared error')
grid on
legend('genie KF','IMM','TCN')
title(['sum of MSE on x and y coordinates over all ' num2str(numSims) ' realizations'])
