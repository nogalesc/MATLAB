function [ rmse ] = loocvreg_slow(xTr,yTr)
%LOOCVREG_SLOW 
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
%% Calculate rmse slowly
tic
% Matrix that hold rmse values at each iteration
m_loocv_error = zeros(n,1);   
% Make temporary X and y matrices
Xtemp = X;
ytemp = y;
for i=1:n
    % Step 1. Train on all data points, except xi
    % Remove the ith sample from temporary X and y
    Xtemp(i,:) = [];
    ytemp(i,:) = [];
    % Calculate w
    w = pinv(Xtemp'*Xtemp)*Xtemp'*ytemp;
    % Use that to get best estimate at all samples (including xi)
    y_hat = X*w;
    % Step 2. Predict yi for that removed sample
    m_loocv_error(i) = (y(i) - y_hat(i))^2;
    % Reset temporary X and y
    Xtemp = X;
    ytemp = y;
end
loocv = sum(m_loocv_error);
rmse = sqrt((1/n)*loocv);
disp('Running loocvreg_slow');
toc