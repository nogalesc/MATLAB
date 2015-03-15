% LOGISTIC REGRESSION CLASSIFIER

clc; clear; close all;
tic

% Set regularization parameter lambda to 1
lambda = 1;
eta = 0.0001;
% load data
load('SenatorVoting.mat')
% Get number of training examples
NumTrainingSamples = size(TrainData,1);
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
% Permute randomly the samples
[XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
[TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
X = TrainD;
Y = TrainL;
u = TestD;
v = TestL;

nf=size(X,2);  % number of features = 543

% Initialize fitting parameters
initial_theta = zeros(nf, 1);     % all_theta = 543x1
% Compute initial cost and gradient
[cost, grad] = costFunctionReg(initial_theta, X, Y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
%  Run fminunc to obtain the optimal theta
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, X, Y,lambda)), initial_theta, options);
% Print cost from theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
% Predict & calculate accuracy on test set
pred = predictLR(theta, u);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == v)) * 100);

toc
