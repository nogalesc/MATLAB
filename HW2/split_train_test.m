function [ TrainData, TrainLabels, TestData, TestLabels ] = split_train_test( InputData, InputLabels )
% Split TrainData into 2/3 for training and 1/3 testing
% ASSUMPTION: samples are column-wise (i.e. 50x542)
n = size(InputData,1);
idx = ceil(n*(2/3));
% Split the data
TrainData = InputData(1:idx,:); 
TrainLabels = InputLabels(1:idx,:);
TestData = InputData(idx+1:end,:);
TestLabels = InputLabels(idx+1:end,:);
end

