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
    [NB_confidence(cur), ~] = trainNB(X,Y,u,v);
    % Logistic Regression
    [LR_confidence(cur), ~] = trainLR(X,Y,u,v,lambda);
end
% Print out the average performance of both algorithms
fprintf('\nNaive Bayes Average Accuracy: %f\n',mean(NB_confidence)*100);
fprintf('\nLogistic Regression Average Accuracy: %f\n',mean(LR_confidence)*100);
%% Problem 4.3.2  Learning Curve
% load data (again)
clear; close all;
load('SenatorVoting.mat')
% Use only [2 4 8 16 33] of these points for learning the
% parameters of NB and LR
num_to_use = [2 4 8 16 33];
% Set regularization parameter lambda to 1
lambda = 1;
% Get number of training examples
NumTrainingSamples = size(TrainData,1);
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
for trial=1:5
    % Permute randomly the samples
    [XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
    [TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
    for cur_test=1:5
        cur_num = num_to_use(cur_test);
        % Extract a certain number of training samples
        if(cur_test ~= 5)
            [X,Y] = Extract_Training_Set(cur_num,TrainD,TrainL);
        else
            X = TrainD;
            Y = TrainL;
        end
        u = TestD;
        v = TestL;
        % Naive Bayes
        [NB_confidence(cur_test), ~] = trainNB(X,Y,u,v);
        % Logistic Regression
        [LR_confidence(cur_test), ~] = trainLR(X,Y,u,v,lambda);
    end
    NB_error(trial) = 1 - mean(NB_confidence);
    LR_error(trial) = 1 - mean(LR_confidence);
end
plot(num_to_use,LR_error,num_to_use,NB_error);
legend('LR error','NB error')
hold on;
% Print out the average performance of both algorithms
fprintf('\nNaive Bayes Average Error: %f\n',NB_error);
fprintf('\nLogistic Regression Average Error: %f\n',LR_error);

