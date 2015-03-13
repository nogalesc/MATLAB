function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
% y = mx1 column vector. Number of training examples = m
m = length(y); 
% J = single number representing cost
J = 0;
% gradient = nx1 column vector (same size as theta)
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% hypothesis = mx1 column vector
hypothesis = sigmoid(X*theta);
% initial cost = J
J = 1./m * ( -y' * log(hypothesis) - (1-y)' * log(1-hypothesis));
grad = 1./m * X' * (hypothesis - y);
% =============================================================

end
