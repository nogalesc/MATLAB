% HW2

% %% Problem 2.8
% clc; clear; close all;
% load('cars.mat')
% slow_rmse = loocvreg_slow(xTr,yTr);
% fast_rmse = loocvreg_fast(xTr,yTr);
%% Problem 4.3
clc; clear; close all;
load('SenatorVoting.mat')
% Step 1. Prepare data into trainining and test sets
% Add intercept term to x and X_test
NumTrainingSamples = size(TrainData,1); % number of training examples
% Add 1 to the TrainData features (543)
% TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
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
%============================================================
% Logistic Regression
% Bias automatically added when calculating thetas
num_labels = 1;
[all_theta] = oneVsAll(TrainData, TrainLabel, num_labels, lambda); % all_theta = 2x543
% Predict using the thetas (w) we found:
pred = predictOneVsAll(all_theta, X);



