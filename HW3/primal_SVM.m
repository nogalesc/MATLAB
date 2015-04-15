% single SVM
clc; clear;
load('hw3_raw_data.mat');      % First extract data
C = 10;
% Extract labels
y = traindata(:,29);
X = traindata;
X(:,29) = [];
ntrain = size(X,1);            % 10,000
ndim = size(X,2);              % 28
sz_z = ndim + 1 + ntrain;      % % size of z = [w w0 ei] => 10,029
% construct H
H = zeros(sz_z);               % 10,029 x 10,029
H(1:ndim,1:ndim) = eye(ndim);
% construct f
f = [zeros(ndim,1); zeros(1); C*ones(ntrain,1)];  % 10,029
% construct A
A = -[(y*ones(1,ndim)).*X  y eye(ntrain)];
% construct b
b = (-1)*ones(size(y));
% construct lb = Represents the lower bounds elementwise in lb ≤ x ≤ ub.
lb = [-inf(1,ndim) -inf zeros(1,ntrain)];

tic
[z err lm] = quadprog(H,f,A,b,[],[],lb);       % THIS ONE WORKED.

% [z err lm] = qpas(H,f,A,b,[],[],lb);       % THIS ONE WORKED.
toc


w = z(1:ndim,1);
w0 = z(ndim+1,1);
% w0 = sum(y - (X*svmModel.w))/ntrain;            % My bias is w0
save('primal_SVM_results_w_w0.mat','z','err','lm','w','w0');

% tic;xqpas=qpas(H,f,L,k,A,b,l,u,dsp);toc
% toc
%                                                                                                                                                                                                                                                                                                                                                                      
%  In Matlab a call to this function would look like:
%
%    [x,err,lm] = qp(H,f,L,k,A,b,l,u,display);
%
%  where the inputs are:
%
%        H: An (n x n) positive semi-definite symmetric matrix
%
%         f: A n element column vector
%   
%    (L,k): General linear inequality constraints
%
%    (A,b): General linear equality constraints
%
%        l: Element-wise lower bound constraints
%
%        u: Element-wise upper bound constraints
%
%  display: If display>0 then iteration information is displayed

%  and the outputs are:
%
%              x: the optimal solution (if obtained)
%
%            err: error number, if err=0, then x is optimal
%
%             lm: structure of Lagrange multipliers
