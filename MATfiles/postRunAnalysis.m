function postDataAnalysis(logFile)
    load(logFile, 'testInfo');
    [~,numTests] = size(testInfo);
    predDataPoints = zeros(numTests,4);
%     estDataPoints = zeros(numTests, 4);
    
    for i=1:numTests
        intermediate = testInfo(i);
        predDataPoints(i,1) = intermediate{1}.predictionMSE;
        predDataPoints(i,2) = intermediate{1}.riccatiConvergencePred;
        predDataPoints(i,3) = intermediate{1}.LS_PredMSE;
        predDataPoints(i,4) = intermediate{1}.KF_PredMSE;
        
%         estDataPoints(i,1) = intermediate{1}.estimationMSE;
%         estDataPoints(i,2) = intermediate{1}.riccatiConvergenceEst;
%         estDataPoints(i,3) = intermediate{1}.LS_EstMSE;
%         estDataPoints(i,4) = intermediate{1}.KF_EstMSE;
    end
    close all;
    figure(1);
    hold on;
    TCN_PMSE = plot(predDataPoints(:,2),predDataPoints(:,1), 'g*');
    LS_PMSE = plot(predDataPoints(:,2),predDataPoints(:,3), 'bo');
    KF_PMSE = plot(predDataPoints(:,2),predDataPoints(:,4), 'r.');
    
    % Drawing 45 degree line for people to see
    x = 0:3;
    plot(x,x,'k--')
    
    xlabel('Riccati Best Possible Prediction MSE')
    ylabel('Other method Prediction MSEs')
    title('Graph of Prediction')
    grid on;
    hold off;
    legend([TCN_PMSE, LS_PMSE, KF_PMSE], 'MSE of TCN', 'MSE of Least Squares', 'MSE of Kalman Filter');
    
%     figure(2);
%     hold on;
%     TCN_EMSE = plot(estDataPoints(:,2),estDataPoints(:,1), 'go');
%     LS_EMSE = plot(estDataPoints(:,2),estDataPoints(:,3), 'bo');
%     KF_EMSE = plot(estDataPoints(:,2),estDataPoints(:,4), 'ro');
%     
%     % Drawing 45 degree line for people to see
%     x = 0:1;
%     plot(x,x,'k--')
%     
%     xlabel('Riccati Best Possible Estimate MSE')
%     ylabel('Other method Estimate MSEs')
%     title('Graph of Estimates')
%     axis([0, 0.1, 0, 0.1])
%     grid on;
%     hold off;
%     legend([TCN_EMSE, LS_EMSE, KF_EMSE], 'MSE of TCN', 'MSE of Least Squares', 'MSE of Kalman Filter');
    
    disp('fin');
end

