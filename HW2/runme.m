% HW2

% %% Problem 2.8
clc; clear; close all;
load('cars.mat')
slow_rmse = loocvreg_slow(xTr,yTr);
fast_rmse = loocvreg_fast(xTr,yTr);
%% Problem 4.3.1 Train LR and NB
clear; close all;
load('SenatorVoting.mat')
%% Set up variables, training set, and test set
% Set regularization parameter lambda to 1
lambda = 1;
% Get number of training examples
NumTrainingSamples = size(TrainData,1);
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
for cur=1:5
    % Permute randomly the samples
    [XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
    [TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
    X = TrainD;
    Y = TrainL;
    u = TestD;
    v = TestL;
    % Naive Bayes
    NB_confidence(cur) = trainNB(X,Y,u,v);
    % Logistic Regression
    LR_confidence(cur) = trainLR(X,Y,u,v,lambda);
end
% Print out the average performance of both algorithms
fprintf('\nNaive Bayes Average Accuracy: %f\n',mean(NB_confidence));
fprintf('\nLogistic Regression Average Accuracy: %f\n',mean(LR_confidence));
%% Problem 4.3.2  Learning Curve
% for split=1:5
% % Use only [2 4 8 16 33] of these points for learning the
% % parameters of NB and LR
% test = [2 4 8 16 33];
% end