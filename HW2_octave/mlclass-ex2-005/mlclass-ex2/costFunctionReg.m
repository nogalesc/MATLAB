function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% Return J (cost) and grad (gradient)
J = 0;
grad = zeros(size(theta));
%=============================================================
% hypothesis = mx1 column vector
hypothesis = sigmoid(X*theta);
% initial cost = J
J = 1./m * ( -y' * log(hypothesis) - (1-y)' * log(1-hypothesis)) + (lambda/(2 * m)).* theta'*theta ;
% Regularize all gradients except for the first gradient (theta_0) because
% that is the bias
% Gradient of the bias (not regularized)
grad = 1./m * X' * (hypothesis - y);
% gradient for every other features (regularized)
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);
% =============================================================

end
