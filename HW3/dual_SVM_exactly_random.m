% dual SVM
clc; clear;
load('hw3_raw_data.mat');                      % First extract data
C = 10;                                        % Choose C

%----------------------------DUAL SVM----------------------------------%
% INPUTS: testdata, traindata, valdata
X = traindata;
X(:,29) = [];                  % Erase labels from data
y = traindata(:,29);
% normalize X
% X = normalize(X); % DOES NOT WORK
X = bsxfun(@rdivide,X,max(X)); % Normalized (scaled) matrix by column
% We have 28 features, 10,000 samples, last col is label
ntrain = size(X,1);                    % 10,000 samples
ndim = size(X,2);                      % 28 features
% construct H
H = zeros(ntrain,ntrain);
for i=1:ntrain
     for j=1:ntrain
         H(i,j) = y(i)*y(j)*X(i,:)*X(j,:)';
     end
end
disp('finished creating H');

f = ones(ntrain,1);                            % construct f = 10000x1 col vector
A = y';                                        % construct A = y? NO. A = y'? YES.
% b = zeros(size(y));                            % construct b = 1x10000
b = 0;
lb = zeros(size(y));                                        % set lb
ub = C*ones(size(y));                                        % set C
% ub = C;
% lb = 0;

% 
% tic
% z = quadprog(H,f,A,b,[],[],lb);                                                               
% z = qpas(H,f,A,b,[],[],lb,ub)
[z err lm] = qpas(H,f,A,b,[],[],lb,ub);       % THIS ONE WORKED.
% save('dual_SVM_results.mat','z','err','lm');
% load('dual_SVM_results.mat');
save('dual_SVM_results_latest_and_greatest.mat','z','err','lm','X','y');
%--qpas errors--%
% ERROR: The L matrix must have 10000 columns.
%    [x,err,lm] = qp(H,f,L,k,A,b,l,u,display);

% ERROR: The k column vector must have 1 entries.
% ERROR: The l column vector must have 10000 entries.



% % w =  sum of all alpha_i's (z) times y_i's time (y) times xi's (x)
% svmModel.w = X'*(z.*y);
% svmModel.w0 = sum(y - (X*svmModel.w))/ntrain;            % My bias is w0
% w = svmModel.w;
% w0 = svmModel.w0;
% save('dual_SVM_w_w0_try3.mat','w','w0','z','err','lm');
