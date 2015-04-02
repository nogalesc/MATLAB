
% =============================================================
% HINT FROM HW3:
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% =============================================================
% HINT FROM Sumbission:
% Hint: You should not regularize theta(1).
% =============================================================
% Test Case #2
% >> costFunction([0.25 0.5 -0.5]', [1 2.2 .89;1 2.4 -.67],[0; 0])
% J =  1.5924
% =============================================================
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
theta(1) = 0;
% regularization
% initial Cost = J
J = (1/m) * ( -y' * log(hypothesis) - (1-y)' * log(1-hypothesis)) + (lambda/(2*m)).*(theta'*theta);
% Regularize all gradients except for the first gradient (theta_0) because
% that is the bias
% Gradient of the bias (not regularized)
grad = 1./m * X' * (hypothesis - y);
% gradient for every other features (regularized)
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);
end

% % Test
% X = [1 2 3 4;-5 -6 -7 -8];
% y = [-2 2]';
% theta = [1 2 -3 -4]';
% [J g] = costFunctionReg(theta, X, y, 1)    % test case with regularization
% J = -30.772
% g = [3.5 ; 6.0 ; 5.0 ; 6.0]
