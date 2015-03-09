function [ g ] = sigmoid( z )
% z can br a matrix
g = (1./(1+exp(-z)));
end

