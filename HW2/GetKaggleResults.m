%% Predict values for Kaggle
clear; close all;
load('SenatorVoting.mat')
% Set regularization parameter lambda to 1
lambda = 1;
% Get number of training examples
NumTrainingSamples = size(TrainData,1);
NumTestSamples = size(TestData,1);
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData];
TestData = [ones(NumTestSamples, 1) TestData]; 
% Permute randomly the samples
[XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
[TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
X = TrainD;
Y = TrainL;
u = TestD;
v = TestL;
[LR_confidence, w] = trainLR(X,Y,u,v,lambda);
predictions = predictLR(w, TestData);
senator_id = [1:49]';
A = num2cell(horzcat(senator_id,predictions));
myCell = {'Senator_ID','Party'};
finalAnswer =vertcat (myCell,A);
fprintf('\nLogistic Regression Accuracy: %f\n', LR_confidence);
% save mat file
save('HW2_Chris_Nogales_predictions.m','finalAnswer','w')
% Uncomment the following line to save data to excel sheet:
% xlswrite('Chris_Nogales.xls',finalAnswer)
        