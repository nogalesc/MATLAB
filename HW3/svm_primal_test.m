% Test SVM dual results
load('hw3_raw_data.mat');    % First extract test data = 10k x 28
load('primal_SVM_results_w_w0.mat');      % Load w and w0

% Verify how well it works on validation data
true_y = valdata(:,29);
valdata(:,29) = [];                  
preds = valdata*w + w0;
% set all negative values to -1 and positive values to +1
n = size(valdata,1);  
% show ROC curve
[X,Y] = perfcurve(true_y,preds,+1);
plot(X,Y);
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by SVM')
% Copy predictions to calculate accuracy on training data 
% Accuracy = # right/ # total
p = preds;
p(p<0) = -1;
p(p>0) = +1;
% Multiply the two vectors. Equal ones will be positive, diff ones are
% negative.
Accuracy_v = p.*true_y;
num_right = sum(Accuracy_v>0);
Accuracy = num_right/n
% Make special matrix of alpha-numeric entries 
my_ID = [1:n];
finalAnswer = horzcat(my_ID',preds);



% % Create csv file for test data
% y = testdata*w + w0;
% n = size(testdata,1);  
% % Make special matrix of alpha-numeric entries 
% my_ID = [1:n];
% predictions = horzcat(my_ID',y);
% A = num2cell(predictions);
% myCell = {'EventID','Prediction'};
% finalAnswer =vertcat (myCell,A);
% % Uncomment the following line to save data to excel sheet:
% % xlswrite('Nogales_Chris_HW3.xls',finalAnswer)