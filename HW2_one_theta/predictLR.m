function [ p ] = predictLR( theta, X )
% Number of Samples
m = size(X, 1);
% Number of Features
n = size(X, 2);

p = X*theta;
p(p>0.5)=1; % or other way?
p(p<=0.5)=0;


end

