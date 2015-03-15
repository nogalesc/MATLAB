% NAIVE BAYES CLASSIFIER

clc; clear; close all;
tic

% Set regularization parameter lambda to 1
lambda = 1;
eta = 0.0001;
% load data
load('SenatorVoting.mat')
% Get number of training examples
NumTrainingSamples = size(TrainData,1); % number of training examples
% Add vector of 1's
TrainData = [ones(NumTrainingSamples, 1) TrainData]; 
% Permute randomly the samples
[XX ,YY] = randomly_permute_both(TrainData, TrainLabel);
[TrainD, TrainL, TestD, TestL ] = split_train_test(XX,YY);
X = TrainD;
Y = TrainL;
u = TestD;
v = TestL;

yu=unique(Y);  % Return the values of the labels {0,1}
nc=length(yu); % number of classes = 2
nf=size(X,2);  % independent variables (features) = 543
ns=length(v);  % number of samples in the test set = 16

% compute class probability using training set
for i=1:nc
    % how many zero(republican) class over the whole?
    % how many one (democrat) class over the whole?
    fy(i)=sum(double(Y==yu(i)))/length(Y);              % 1x2 vector
end
% normal distribution parameters using training set
for i=1:nc
    % extract rows that match a certain class, store in xi
    xi=X((Y==yu(i)),:);         % temporary matrix
    % get the mean and variance of each feature
    mu(i,:)=mean(xi,1);         % 2x543 matrix
    sigma(i,:)=std(xi,1);       % 2x543 matrix
end
% probability for test set
for j=1:ns
    % For each candidate, and for each class, returns the normal
    % probability
    fu=normcdf(ones(nc,1)*u(j,:),mu,sigma);
    % multiply all probabilities from each feature (row-wise)
    P(j,:)=fy.*prod(fu,2)';     % 16x2 matrix
end
% get predicted output for each sample in the test set
[~,id]=max(P,[],2);           % index of maximum probability
for i=1:length(id)
    pv(i,1)=yu(id(i));        % convert from index to {0,1}
end
confidence=sum(pv==v)/length(pv);
fprintf('\nTest Set Accuracy: %f\n',confidence * 100);
toc