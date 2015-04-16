function getROC( LABELS, SCORES, POSCLASS )
%Create ROC curve

load fisheriris
x = meas(51:end,1:2);
% Iris data, 2 classes and 2 features
y = (1:100)'>50;
% Versicolor = 0, virginica = 1
b = glmfit(x,y,'binomial');
% Logistic regression
p = glmval(b,x,'logit');
% Fit probabilities for scores
[X,Y,T,AUC] = perfcurve(species(51:end,:),p,'virginica');
plot(X,Y)
xlabel('False positive rate'); ylabel('True positive rate')
title('ROC for classification by SVM')

end


% [X,Y,T,AUC] = PERFCURVE(LABELS,SCORES,POSCLASS) also returns pointwise
%   confidence bounds for the computed values X, Y, T, and AUC if you
%   supply cell arrays for LABELS and SCORES or set NBOOT to a positive
%   integer. To compute the confidence bounds, PERFCURVE uses either
%   vertical averaging (VA) or threshold averaging (TA). The returned
%   values Y are an M-by-3 array in which the 1st element in every row
%   gives the mean value, the 2nd element gives the lower bound and the 3rd
%   element gives the upper bound. The returned AUC is a row-vector with 3
%   elements following the same convention. For VA, the returned values T
%   are an M-by-3 array and X is a column-vector. For TA, the returned
%   values X are an M-by-3 matrix and T is a column-vector. 
%
