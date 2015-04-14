% dual SVM
clc; clear;
load('hw3_raw_data.mat');                      % First extract data
C = 10;                                        % Choose C
% We have 28 features, 10,000 samples, last col is label
ntrain = size(traindata,1);                    % matrix 10000x28
ndim = size(traindata,2);
%----------------------------DUAL SVM----------------------------------%
% INPUTS: testdata, traindata, valdata
my_testdata = testdata(1:10,:);
my_traindata = traindata(1:10,:);
my_ntrain = size(my_traindata,1);                
my_ndim = size(my_traindata,2);
y = my_traindata(:,29);
my_traindata(:,29) = [];
% for i=1:my_ntrain
%     for j=1:my_ndim
%     end
% end

H = (y*y')*(my_traindata*my_traindata');


% tic
% z = quadprog(H,f,A,b,[],[],lb);
% toc
% 

