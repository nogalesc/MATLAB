function [ TrainD, TestD ] = split_train_test( TrainData )
% Split TrainData into 2/3 for training and 1/3 testing
% ASSUMPTION: samples are column-wise (i.e. 50x542)
n = size(TrainData,2);
idx = ceil(n*(2/3));
TrainD = TrainData(:,(1:idx)); 
TestD = TrainData(:,(idx+1:end));
end

