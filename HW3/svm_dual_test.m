% Test SVM dual results
load('hw3_raw_data.mat','testdata');    % First extract test data = 10k x 28
load('dual_SVM_w_w0_try3.mat');      % Load w and w0
y = testdata*w + w0;
n = size(testdata,1);  
% Make special matrix of alpha-numeric entries 
my_ID = [1:n];
predictions = horzcat(my_ID',y);
A = num2cell(predictions);
myCell = {'EventID','Prediction'};
finalAnswer =vertcat (myCell,A);
% Uncomment the following line to save data to excel sheet:
xlswrite('Nogales_Chris_HW3_try3.xls',finalAnswer)