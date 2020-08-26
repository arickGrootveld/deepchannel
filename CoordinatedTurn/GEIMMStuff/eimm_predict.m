function [X_p,P_p,c_j,X,P] = eimm_predict(X_ip,P_ip,w,p_ij,ind,dims,F,Q,dt)
    % Number of models 
    m = length(X_ip);
    
    % Default values for state mean and covariance
    MM_def = zeros(dims,1);
    PP_def = diag(20*ones(dims,1));

    % Normalizing factors for mixing probabilities
    c_j = zeros(1,m);
    for j = 1:m
        for i = 1:m
            c_j(j) = c_j(j) + p_ij(i,j).*w(i);
        end
    end
    
    %if(c_j(1) == 0 || c_j(2) == 0)
    %    c_j = [0.00000001, 0.0000001];
    %end
    
    % Mixing probabilities
    MU_ij = zeros(m,m);
    for i = 1:m
        for j = 1:m
            MU_ij(i,j) = p_ij(i,j) * w(i) / c_j(j);
        end
    end

    % Calculate the mixed state mean for each filter
    X_0j = cell(1,m);
    for j = 1:m
        X_0j{j} = zeros(dims,1);
        for i = 1:m
            X_0j{j}(ind{i}) = X_0j{j}(ind{i}) + X_ip{i}*MU_ij(i,j);
        end
    end
    
    % Calculate the mixed state covariance for each filter
    P_0j = cell(1,m);
    for j = 1:m
        P_0j{j} = zeros(dims,dims);
        for i = 1:m
            P_0j{j}(ind{i},ind{i}) = P_0j{j}(ind{i},ind{i}) + MU_ij(i,j)*(P_ip{i} + (X_ip{i}-X_0j{j}(ind{i}))*(X_ip{i}-X_0j{j}(ind{i}))');
        end
    end

    % Space for predictions
    X_p = cell(1,m);
    P_p = cell(1,m);

    % Make predictions for each model
    for i = 1:m
        [X_p{i}, P_p{i}] = ekf_predict(X_0j{i}(ind{i}),P_0j{i}(ind{i},ind{i}),F{i},Q{i},dt);
    end

    % Output the combined predicted state mean and covariance, if wanted.
    if nargout > 3
        % Space for estimates
        X = zeros(dims,1);
        P = zeros(dims,dims);
        
        % Predicted state mean
        for i = 1:m
            X(ind{i}) = X(ind{i}) + w(i)*X_p{i};
        end

        % Predicted state covariance
        for i = 1:m
            P(ind{i},ind{i}) = P(ind{i},ind{i}) + w(i)*(P_p{i} + (X_ip{i}-X(ind{i}))*(X_ip{i}-X(ind{i}))');
        end
    end
end
    