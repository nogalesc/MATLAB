function [ rmse ] = loocvreg_fast(xTr,yTr)
%LOOCVREG_FAST 
%   This function calculates the leave-on-out 
%   cross-validation rmse, or
%   'root mean squared error' (rmse) 
%% Set up variables (using column vectors)
% xTr 6 features x 93 samples
% yTr 1x93 labels
% X 93 samples x 6 features 
% y 93x1 labels
X = xTr';                                  
y = yTr';   
% Number of Samples n = 93
n = size(y,1);
% Add bias feature (ones) to each sample
% X 93 samples x 7 features 
X = [ones(n,1) X];
%% Calculate rmse fast
% Matrix that hold rmse values at each iteration
m_loocv_error = zeros(n,1);    
% Calculate the H matrix
H = X*pinv(X'*X)*X';
% Return column vector of the main diagonal elements of H.
Hkk = diag(H);
% Calculate y_hat
y_hat = H*y;
% The rmse error is calculated for each time you remove a sample.
for k=1:n   
    m_loocv_error(k) = ((y(k) - y_hat(k))/(1-Hkk(k)))^2;
end
% Add up all the squared errors to the loocv error variable
loocv = sum(m_loocv_error);
rmse = sqrt((1/n)*loocv);

