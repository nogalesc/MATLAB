% HW2

%% Problem 2.8
clc
clear
load('cars.mat')
slow_rmse = loocvreg_slow(xTr,yTr)
fast_rmse = loocvreg_fast(xTr,yTr)
%% Problem 4.3
load('SenatorVoting.mat')
% Step 1. Prepare data into trainining and test sets
% Add intercept term to x and X_test
NumTrainingSamples = size(TrainData,1); % number of training examples
% Add 1 to the TrainData features (543)
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
% Randomly permute data and split it into training and test data
[XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
[TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
% Step 2. Train Logistic Regression classifier
% Set regularization parameter lambda to 0.001
lambda = 0.001;
eta = 0.0001;
m = size(TrainD,1);
% Initialize fitting parameters
param_sz = size(TrainD, 2);
initial_w = zeros(param_sz, 1);
J = 0;
w = initial_w;
X = TrainD;
y = TrainL;
% hypothesis:
h = sigmoid(X*w);
% Update w until convergence:
tol = 100;
prev_w = w;
while tol>.00001  % *desired accuracy*
    prev_w = w;
    w = w + eta*X'*(y-h) - eta*lambda*w;
    tol = abs(w - prev_w);
end
% Compute and display initial cost and gradient for regularized logistic
% regression
% [cost, grad] = costFunctionReg(initial_w, TrainD, TrainL, lambda);
%=============================================================

% J = (1/m)*sum(-y'.*log(h)-(1-y)'.*log(1-h)) + (lambda/(2*m))*(sum(w.^2)-w(1,1)^2);

% w_1 = w;
% w_1(1) = 0;
% J = 1/m * ( -y' * log(sigmoid(X*w)) - (1-y)' * log(1 - sigmoid(X*w))) + lambda/(2*m) * sum (w_1 .* w_1);
% grad = (X' * (sigmoid(X*w)-y) + lambda .* w_1)./ m;

% % calculate initial cost:
% J = (1/m)*sum(-y'.*log(h)-(1-y)'.*log(1-h)) + (lambda/(2*m))*(sum(w.^2)-w(1,1)^2);
% grad(1,1) = (1/m)*sum((sigmoid(X*w)-y).*X(:,1));
% for i=2:length(w)
% grad(i,1) = (1/m)*sum((h)-y).*X(:, i)) + lambda*w(i,1)/m;
% end
% fprintf('Cost at initial w (zeros): %f\n', cost);
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;
% 
% w_hat = pinv(TrainD)*TrainL;     
% Y_hat = X*w_hat;
% 
% %     sigmoid.m 
%     costFunction.m
%     predict.m
%     costFunctionReg.m
