function [ confidence ] = trainLR(X,Y,u,v,lambda)
    nf=size(X,2);  % number of features = 543
    % Initialize fitting parameters
    initial_theta = zeros(nf, 1);     % all_theta = 543x1
    % Compute initial cost and gradient
    % [cost, grad] = costFunctionReg(initial_theta, X, Y, lambda);
    % fprintf('Cost at initial theta (zeros): %f\n', cost);
    %  Run fminunc to obtain the optimal theta
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = ...
        fminunc(@(t)(costFunctionReg(t, X, Y,lambda)), initial_theta, options);
    % Print cost from theta to screen
    % fprintf('Cost at theta found by fminunc: %f\n', cost);
    % Predict & calculate accuracy on test set
    pred = predictLR(theta, u);
    confidence = mean(double(pred == v)) * 100;
    % fprintf('\nLR Test Set Accuracy: %f\n',confidence);
end