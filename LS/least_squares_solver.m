clc
clear
load('./ARMismatchPredictor/data/data1.mat');
% load('./ARProcessPredictor/logs/log1.mat');
N = 30;
M = 50;

[batchSize, ~, seqLength, seriesLength] = size(measuredData);

measuredConvertedData = zeros(batchSize*seriesLength, seqLength);

for i=1:seriesLength
    for m=1:batchSize
        measuredConvertedData(m+((i-1)*batchSize), :) = ... 
            measuredData(m,1,:,i) + measuredData(m,2,:,i)*1j;
    end
end

state = 1;
Xhat = squeeze(predAndCurState(1,state,:));
Y1 = squeeze(trainDataMeas(1:size(Xhat,1), state));



Xhat1 = Xhat(1:M, state);
Ymat = zeros(M, N+1);

k = 1;
for j = 1: N+1
    Ymat(:,j) = Y1(k+1:k+M);
    k = k+1;
end

a_ls = (Ymat'*Ymat)\(Ymat'*Xhat1); 
MSE = 1/(M+1)*(Xhat1 - Ymat*a_ls)'*(Xhat1 - Ymat*a_ls);