% % dual SVM
% clc; clear;
load('hw3_raw_data.mat');                      % First extract data
C = 100;                                        % Choose C

% %----------------------------DUAL SVM----------------------------------%
% INPUTS: testdata, traindata, valdata
X = traindata;
X(:,29) = [];                  % Erase labels from data
y = traindata(:,29);
% normalize X
% X = bsxfun(@rdivide,X,max(X)); % Normalized (scaled) matrix by column
% % We have 28 features, 10,000 samples, last col is label
ntrain = size(X,1);                    % 10,000 samples
ndim = size(X,2);                      % 28 features
% % construct H
% H = zeros(ntrain,ntrain);
% for i=1:ntrain
%      for j=1:ntrain
%          H(i,j) = -y(i)*y(j)*X(i,:)*X(j,:)';
%      end
% end
% save('negative_H.mat','H')
% disp('finished creating H');
load('negative_H.mat')
H = -H;
% Other variables
f = (-1)*ones(size(y));                            % construct f = 10000x1 col vector
A = -1*eye(ntrain);                                        % construct A = y? NO. A = y'? YES.
b = zeros(size(y));                           % construct b = 1x10000
% b = 0;
% Aeq = ?;
% beq = 0;
lb = zeros(size(y));                                        % set vector of doubles lb
ub = C*ones(size(y));                                       % set vector of coubles C
% Aeq(1,:) = y';
% beq = 0; 
Aeq = zeros(ntrain);
Aeq(1,:) = y';
beq = zeros(size(y));

tic  
% alphas = quadprog(H,f,A,b,Aeq,beq,lb,ub,[]); % This worked
alphas = qpas(H,f,A,b,Aeq,beq,lb,ub,[]); % This worked
%    qpas (Aeq,beq): General linear equality constraints
toc

save('dual_SVM_result_12am.mat','alphas','X','y');

% Elapsed time is 1189.497514 seconds.

% Save results:
% Elapsed time is 1265.487561 seconds.

% w = X'*(alpha.*y);
% w0 = sum(y - (X*w))/ntrain;            

% Error:
% The number of columns in A must be the same as the number of elements of f.

%  where the inputs are:
%
%        H: An (n x n) positive semi-definite symmetric matrix
%
%	     f: A n element column vector
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
