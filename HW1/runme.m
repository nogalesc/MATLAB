% Nearest Neighbor script to run knncv.m and knn.m


% Predict labels
%% Obtain data and set variables
clc
clear
%{
% There are 800 training images.
% From each image, they took 512 GIST features
% For each image, there is a label (1-8)
  k = number of neighbors
  f = distance function to use (ex: 'sqeuclidean')
  D = Matrix (ntrain x ntest) 
 %}
% Add path, load data
addpath('/home/nogalesc/MATLAB/HW1/k-NN/k-NN')
S = load('SceneCateg.mat')
traindata = S.trainfeatgist;
trainlabels = S.trainlabels;
testdata = S.testfeatgist;
% Choose k,f,n (D is optional)
k = 11
f = 'sqeuclidean';
n = 10;                              % 10 fold cross validation
D = distEucSq(traindata, testdata);  % Hint: precompute the distances b/t all pairs of points
%Im = mat2gray(D);
%imshow(Im)
%% Problem 4.2.1
predlabels = knn(traindata, trainlabels, testdata, k, f);
% Make special matrix of alpha-numeric entries 
Image_ID =reshape(1:size(predlabels,2),1,size(predlabels,2));
predictions = horzcat(Image_ID',predlabels');
A = num2cell(predictions)
myCell = {'Image_ID','Category'};
finalAnswer =vertcat (myCell,A);
% Save data to excel sheet
xlswrite('test_finalAnswer.xls',finalAnswer)
%% Problem 4.2.2
cv_error = knncv(traindata, trainlabels, n, k, f, D)

