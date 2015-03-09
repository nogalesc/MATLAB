function [ rmse ] = loocvreg_slow(xTr,yTr)
%LOOCVREG_SLOW 
%   This function calculates the leave-on-out 
%   cross-validation rmse, or
%   'root mean squared error' (rmse) 
%   xTr = 6x93
%   yTr = 1x93
%% Set up variables (using column vectors)
n = size(xTr,2);               % sample size
vertcat(ones(1,n), xTr);       % xTr = 7x93
X = xTr';                      % X = 93x7
Y = yTr';                      % Y = 93x1
m_rmse = zeros(size(yTr'));    
%% Naive LOOCV
for k=1:size(xTr,2)                        % run 93 times
    k
    X(k,:) = [];                           % delete kth
    Y(k,:) = [];           
    H = X*pinv(X);     
    Y_hat = H*Y;
    loocv_error = sum((Y - Y_hat).^2);
    m_rmse(k) = sqrt((1/n)*loocv_error)
    X = xTr';                               % add kth back
    Y = yTr';
end
% %% Nested fuctions
% % Calculate squared difference
% function sum_sqrd_error = sum_of_sqrd_diff(y,y_hat)
%   sum_sqrd_error = (y - y_hat).^2; 
% end
rmse = min(m_rmse);
end



    %pinv
    % y = H*y