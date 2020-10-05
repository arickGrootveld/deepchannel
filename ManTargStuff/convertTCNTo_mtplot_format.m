load('../TCN/logs/log372.mat');

% errs = testInfo{1,1}.instantaneousSquaredErrors;
YbadForm = testInfo{1,1}.tcnPredVals;

s3 = size(YbadForm, 3);
s2 = size(YbadForm, 2);

YY = zeros(s2*s3, 2);

% Jamming predicted values into a matrix so we can actually compare 
% the TCN and the IMM
for i = 0:s3-1
    YY(((i*s2)+1):((i+1)*s2), :) = YbadForm(:, :, i+1)';
end

save('test6_tcn.mat', 'YY');
