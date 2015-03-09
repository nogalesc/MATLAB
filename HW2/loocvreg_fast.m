function [ rmse ] = loocvreg_fast(xTr,yTr)
%LOOCVREG_FAST 
%   This function calculates the leave-on-out 
%   cross-validation rmse, or
%   'root mean squared error' (rmse) 
%   xTr = 6x93
%   yTr = 1x93
%% Set up variables (using column vectors)
n = size(xTr,2);                           % sample size
xTr = vertcat(ones(1,n), xTr);             % xTr = 7x93
X = xTr';                                  % X = 93x7
Z = yTr';                                  % Z = 93x`
Y = yTr';                                  % Y = 93x1

m_rmse = zeros(size(yTr'));    
%% fast LOOCV
for k=1:size(xTr,2)                        % run 93 times
    H = X*pinv(X);                         % get H before removing kth
%     Y(k,:) = [];                         % remove kth?
%     n = size(Y,1);
    loocv_error = sum(((Y(k) - H(k,k)*Y(k))/(1-H(k,k))).^2);
    m_rmse(k) = sqrt((1/n)*loocv_error);
    Y = yTr';
end
rmse = min(m_rmse);
end