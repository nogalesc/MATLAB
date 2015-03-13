% g1 = [1 1 2 2 3 3]';		% Known groups
% g2 = [1 1 2 3 4 NaN]';	% Predicted groups
% 
% [C,order] = confusionmat(g1,g2)

%read data example: Import columns as column vectors 
%[W X Y Z] = csvimport('shadwell_2014_11_06_04_33_.csv', 'columns', {'Predicted_class', 'Predicted', 'Truth_class','Truth'});
%[W X Y Z] = xlsread('Shadwelll_Confusion_matrix_2014_11_06_2015-04-33_comparison _of_classes.csv');
clc
clear
my_matrix = xlsread('Shadwelll_Confusion_matrix_2014_11_06_2015-04-33_comparison _of_classes.csv');
% Assumption: 
% First column = Predicted
% Second column = Ground Truth
predicted_vector = my_matrix(:, 1);       
ground_vector = my_matrix(:, 2);       
[C,order] = confusionmat(ground_vector,predicted_vector) 
% By order of FIRE, FIRE-R
filename = 'shadwell_demo_comparison_2014_11_06_2015-04-33.mat';
save(filename,'my_matrix')

