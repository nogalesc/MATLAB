function cv_error = knncv(traindata, trainlabels, n, k, f, D)
      % NUMBER OF CLASSES = 8 (labels in 'trainlabels' for 800)
      % NUMBER OF NEAREST NEIGHBORS = k
      % NUMBER OF TRAINING EXAMPLES = 800
      % NUMBER OF FEATURES = 512
      % NUMBER OF PARTITIONS OF TRAINING LABLES = n
%% Partition training data into n parts of ~approx equal size
[divided_D,divided_L] = divide_training_into_n(traindata,trainlabels,n);
% Useful for checking equality of multi-dim array to original array.
if(divided_D(:,:,2) == traindata(81:160,:))
    disp('both equal');
end
if(divided_D(:,:,1) == traindata(1:80,:))
    disp('both equal');
end
% Step 1. Use n-1 of these parts as training data
% Step 2. Make predictions on the held-out part
% Step 3. Repeat for all held out folds. Call the error of these predictions our cross-val error.

 for ROUND = 1:n           
    % Take out desired fold
    cur_testdata = divided_D(:,:,ROUND);
    cur_testlabels = divided_L(:,:,ROUND);
    cur_traindata = giant_matrix_but_one(divided_D,ROUND);
    cur_trainlabels = giant_matrix_but_one(divided_L,ROUND);
    predlabels = knn(cur_traindata, cur_trainlabels, cur_testdata, k, f);
    cur_error = 0;
%     cur_error = calculate_error(predlabels,cur_trainlabels);
    disp(['ROUND = ' num2str(ROUND) '  Error this round = ' num2str(cur_error)]);
 end

cv_error = 99;
% Plot train error: 
% i.e. if you train on all 800 images, and make predictions on these same 800
%images

    % Nested functions
    function B = giant_matrix_but_one(A, delete_me)
       % Delete the i'th matrix
       A(:,:,delete_me)=[];
       % Get row, column, and depth size
       r = size(A,1);
       c = size(A,2);
       p = size(A,3);
       % Create a giant matrix that stacks dimensions vertically
       B = permute(A,[1 3 2]);
       B = reshape(B,{},size(A,2),1);
    end

    
    function error = calculate_error(prediction_in, truth_in)
        num_missed = sum(prediction_in ~= truth_in);
        error = num_missed/size(truth_in,1);
    end

    function [Multi_A_test,Multi_L] = divide_training_into_n(T_data, L_data, n)
      % Assumption 1: Number of rows is the number of training examples
      % Assumption 2: Number of rows is evenly divisible by n.
      div = size(T_data,1)/n;
%       Multi_A = reshape(T_data,div,size(T_data,2),n); % 80x512x10
      diff = div - 1;
      % Create multidimensional array                  
      Multi_A_test = zeros(div,size(T_data,2),n);          % 80x512x10
      for j = 1:n
          up = j*div;
          low = up-diff;
%           disp(['low = ' num2str(low) '  up = ' num2str(up)]); 
          cur_matrix = T_data(low:up,:);
          size(cur_matrix);
          Multi_A_test(:,:,j) = cur_matrix;
      end
      % Also separate the labels
      Multi_L = reshape(L_data,div,1,n);
    end
end




%   % Nested fuctions
%     function [Multi_A,Multi_L] = divide_training_into_n(T_data, L_data, n)
%       % Assumption 1: Number of rows is the number of training examples
%       % Assumption 2: Number of rows is evenly divisible by n.
%       div = size(T_data,1)/n;
%       diff = div - 1;
%       % Create multidimensional array                  
%       Multi_A = zeros(div,size(T_data,2),n);    % 80x512x10
%       for j = 1:n
%           up = j*div;
%           low = up-diff;
% %           disp(['low = ' num2str(low) '  up = ' num2str(up)]); 
%           cur_matrix = T_data(low:up,:);
%           size(cur_matrix);
%           Multi_A(:,:,j) = cur_matrix;
%       end
%       % Also separate the labels
%       Multi_L = reshape(L_data,div,1,n);
% 
%     end

