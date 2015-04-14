% dual SVM
clc; clear;
load('hw3_raw_data.mat');                      % First extract data
C = 10;                                        % Choose C
% We have 28 features, 10,000 samples, last col is label
ntrain = size(traindata,1);                    % matrix 10000x28
ndim = size(traindata,2);
%----------------------------DUAL SVM----------------------------------%
% INPUTS: testdata, traindata, valdata
X = traindata;
y = traindata(:,29);
my_H = (y*y')*(X*X');             % construct H
H = zeros(ntrain,ntrain);
for i=1:ntrain
     for j=1:ntrain
         H(i,j) = y(i)*y(j)*X(i,:)*X(j,:)';
     end
end

% f = ones(1,ntrain);                            % construct f
% A = y*y';                                      % construct A
% b = zeros(size(y));                            % what is b?
% lb = 0;                                        % set lb
% ub = C;                                        % set C
% 
% tic
% z = quadprog(H,f,A,b,[],[],lb);
% z = quadprog(H,f,A,b,[],[],lb,ub)
% toc
% 

