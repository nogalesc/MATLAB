% SVM PRIMAL 
clc; clear;
% Load traindata, valdata, and testdata
load('hw3_raw_data.mat');              

% Train
trainlabels = traindata(:,29);
traindata(:,29) = [];
C = 10;
svmModel = trainSVMprimal(traindata,trainlabels,C);

% Test SVM model results
w = svmModel.w;
w0 = svmModel.w0;
y = testdata*w + w0;

% Save to excel sheet 
n = size(testdata,1); 
my_ID = [1:n];
predictions = horzcat(my_ID',y);
A = num2cell(predictions);
myCell = {'EventID','Prediction'};
finalAnswer =vertcat (myCell,A);
% Uncomment the following line to save data to excel sheet:
% xlswrite('Nogales_Chris_HW3.xls',finalAnswer)