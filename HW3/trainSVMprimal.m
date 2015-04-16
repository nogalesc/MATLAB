function [ svmModel ] = trainSVMprimal(traindata,trainlabels,C)
%TrainSVMPrimal This function takes in traindata, trainlabels and C
%   then it calculates a soft-margin SVM using the primal form
%   OUTPUT: svmModel with w and w0

% set up X and y
y = trainlabels;
X = traindata;
ntrain = size(X,1);                               % 10,000
ndim = size(X,2);                                 % 28
sz_z = ndim + 1 + ntrain;                         % size of z = [w w0 ei] => 10,029
% construct H
H = zeros(sz_z);                                  % 10,029 x 10,029
H(1:ndim,1:ndim) = eye(ndim);
% construct f
f = [zeros(ndim,1); zeros(1); C*ones(ntrain,1)];  % 10,029
% construct A
A = -[(y*ones(1,ndim)).*X  y eye(ntrain)];
% construct b
b = (-1)*ones(size(y));
% construct (lb) lower bounds for z = [w w0 ei]
lb = [-inf(1,ndim) -inf zeros(1,ntrain)];
% use quadratic programming
tic
[z err lm] = quadprog(H,f,A,b,[],[],lb);          
toc
svmModel.w = z(1:ndim,1);
svmModel.w0 = z(ndim+1,1);

end

