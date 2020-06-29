function [X_i,P_i,w,X,P] = eimm_update(X_p,P_p,c_j,ind,dims,Y,H,R)
    % Number of models 
    m = length(X_p);

    % Space for update state mean, covariance and likelihood of measurements
    X_i = cell(1,m);
    P_i = cell(1,m);
    lambda = zeros(1,m);

    % Update for each model
    for i = 1:m
        % Update the state estimates
        if isa(H{i},'function_handle')
            [X_i{i}, P_i{i}, ~, ~, lambda(i)] = ekf_update(X_p{i},P_p{i},Y,H{i},R{i});
        else
            [X_i{i}, P_i{i}, ~, ~, lambda(i)] = kf_update(X_p{i},P_p{i},Y,H{i},R{i});
        end
    end
    
    % Calculate the model probabilities
    w = zeros(1,m); 
    c = sum(lambda.*c_j);
    w = c_j.*lambda/c;
    
    % Output the combined updated state mean and covariance, if wanted.
    if nargout > 3
        % Space for estimates
        X = zeros(dims,1);
        P = zeros(dims,dims);
        % Updated state mean
        for i = 1:m
            X(ind{i}) = X(ind{i}) + w(i)*X_i{i};
        end
        % Updated state covariance
        for i = 1:m
            P(ind{i},ind{i}) = P(ind{i},ind{i}) + w(i)*(P_i{i} + (X_i{i}-X(ind{i}))*(X_i{i}-X(ind{i}))');
        end
    end
end
    