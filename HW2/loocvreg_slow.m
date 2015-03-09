function [ rmse ] = loocvreg_slow(xTr,yTr)
%LOOCVREG_SLOW 
%   This function calculates the leave-on-out 
%   cross-validation rmse, or
%   'root mean squared error' (rmse) 
%   xTr = 6x93
%   yTr = 1x93
%% Set up variables (using column vectors)
n = size(xTr,2);                           % sample size
xTr = vertcat(ones(1,n), xTr);             % xTr = 7x93
X = xTr';                                  % X = 93x7
Y = yTr';                                  % Y = 93x1
m_rmse = zeros(size(yTr'));    
%% Naive LOOCV
for k=1:size(xTr,2)                        % run 93 times
    X(k,:) = [];                           % delete a kth sample
    Y(k,:) = [];      
    cur_n = size(X,1);                     % update n
    w_hat = pinv(X)*Y;     
    Y_hat = X*w_hat;
    loocv_error = sum((Y - Y_hat).^2);
    m_rmse(k) = sqrt((1/cur_n)*loocv_error);
    X = xTr';                               % add kth back
    Y = yTr';
end
rmse = min(m_rmse);
end