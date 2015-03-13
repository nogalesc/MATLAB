load('shadwell_demo_comparison_2014_11_06_2015-04-33.mat') 
% First column = Predicted
% Second column = Ground Truth
predicted_vector = my_matrix(:, 1);       
ground_vector = my_matrix(:, 2);       
% 4 = FIRE, 5 = FIRE-R
[C,order] = confusionmat(ground_vector,predicted_vector) 