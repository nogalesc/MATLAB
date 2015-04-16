% Test SVM dual results
load('hw3_raw_data.mat');    % First extract test data = 10k x 28
% load('dual_SVM_results_latest_and_greatest.mat');      % Load z,X,y
all_data = load('dual_SVM_result_11am.mat');
X = all_data.X;
y = all_data.y;
alpha = all_data.alphas;
% Anything below and above the thresholds do not count
p(p<=C) = 0;
p(p>=0) = 0;
C = 100; %(I think)
% Use KKT conditions to obtain w, w0
ntrain = size(X,1);
w = X'*(alpha.*y);
w0 = sum(y - (X*w))/ntrain;            % My bias is w0

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
% Multiply the two vectors. 
% Equal ones will be positive, diff ones are negative.
Accuracy_v = p.*true_y;
num_right = sum(Accuracy_v>0);
Accuracy = num_right/n
% Make special matrix of alpha-numeric entries 
my_ID = [1:n];
finalAnswer = horzcat(my_ID',preds);
