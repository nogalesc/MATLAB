% Turn this in
% HW2
% 

% %% Problem 2.8
% clc; clear; close all;
% load('cars.mat')
% slow_rmse = loocvreg_slow(xTr,yTr);
% fast_rmse = loocvreg_fast(xTr,yTr);
%============================================================
%% Problem 4.3
clc; clear; close all;
load('SenatorVoting.mat')
% Use only [2 4 8 16 33] of these points for learning the
% parameters of NB and LR
test = [2 4 8 16 33];
%============================================================
% Get number of training examples
NumTrainingSamples = size(TrainData,1);
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
% Permute randomly the samples
[XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
[TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
X = TrainD;
y = TrainL;
% Set regularization parameter lambda to 1
lambda = 1;
eta = 0.0001;
%============================================================
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);     % all_theta = 543x1
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
% ============= Optimizing using fminunc  =============
% Ok, now find the best thetas
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunctionReg(t, X, y,lambda)), initial_theta, options);
% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
% Predict
pred = predictLR(theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
%============================================================
% Naive Bayes


%============================================================
% % Logistic Regression
% % Bias automatically added when calculating thetas
% num_labels = 1;
% [all_theta] = oneVsAll(TrainD, TrainL, num_labels, lambda); 
% % Predict using the thetas (w) we found:
% pred = predictOneVsAll(all_theta, X);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
%============================================================




