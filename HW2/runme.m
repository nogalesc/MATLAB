% HW2

%% Problem 2.8
clc
clear
load('cars.mat')
slow_rmse = loocvreg_slow(xTr,yTr)
fast_rmse = loocvreg_fast(xTr,yTr)
%% Problem 4.3
load('SenatorVoting.mat')
% Step 1. Prepare data
X = TrainData;
Y = TrainLabel;
[XX ,YY ] = randomly_permute_both(X, Y);
[TrainD, TestD ] = split_train_test(XX);
[TrainL, TestL ] = split_train_test(YY);
% Step 2. Train Logistic Regression classifier
lambda = 0.001;

w_hat = pinv(TrainD)*TrainL;     
Y_hat = X*w_hat;

%     sigmoid.m 
%     costFunction.m
%     predict.m
%     costFunctionReg.m
