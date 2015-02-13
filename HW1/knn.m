

function predlabels = knn(traindata, trainlabels, testdata, k, f)
predlabels = size(trainlabels);

switch f
  case char('euclidean')
      fprintf('Using euclidean distance.')
  case char('sqeuclidean')
      fprintf('Using squared euclidean distance.')
      D = distEucSq(traindata, traindata);                 % 800x800
      sum_of_D = sum(D);
      % Sort column-wise each of the 800 columns
      [k_distances, index] = sort(D,'ascend');
      % Delete distances after k
      index = delete_after_k(index);
      % a = Matrix of classes for those k neighbors
      a = index;
      a(a>0 & a<=100) = 1;
      a(a>100 & a<=200) = 2;
      a(a>200 & a<=300) = 3;
      a(a>300 & a<=400) = 4;
      a(a>400 & a<=500) = 5;
      a(a>500 & a<=600) = 6;
      a(a>600 & a<=700) = 7;
      a(a>700 & a<=800) = 8;
      % find matlab range of values and replace with class #
      % find(1-100) => class 1
      % find(101-200) => class 2
      % find(201-300) => class 3
      % find(301-400) => class 4
      % find(401-500) => class 5 ...
      % find(701-800) => class 8
      k_distances = delete_after_k(k_distances);
      % ---Now I want you to tell me what is the most common class
      % ---in each column--%
      classification_for_each_training_example = mode(a,1);
      % For each near neighbor. Fill in their class.
      
      % NUMBER OF CLASSES = 8 (labels in 'trainlabels' for 800)
      % NUMBER OF NEAREST NEIGHBORS = k
      % NUMBER OF TRAINING EXAMPLES = 800
      % NUMBER OF FEATURES = 512
      % ---Now let's predict the labels for the test data --%
      
      % Plot each of the 100 thingies.
      
      
      % Plot for each of the 800 columns, the k nearest neighbors (by
      % distance)
%     plot(k_distances);
% X — Data to displaymatrix
% Y — Data to plot against Xmatrix
      
      
      several_histograms = histcounts(k_distances,8);
      histogram(D,size(D,2))
      % Create matrix of half Gaussian values
      w = gausswin(k*2)
      nw = cut_in_half(w);
      % Repeat this half Gaussian
      nw = repmat(nw,1,size(index,2));
      
      % No weights, k nearest neighbor
      % Just count...and take the class of the
      % most common neighbor.
      % k = number of neighbors to look at.
      % 
      why = histcounts(D,size(D,2));
      histogram(D,size(D,2))
      % Weighted majority vote
      % (Closest neighbors get more weight.)
      % TRAIN DATA = 800x512 features
      %predlabels = nw*
     
      
      fprintf('Using squared euclidean distance.')
  otherwise
      disp('Error, no distance function! Try again!') 
end
  % Nested fuctions
    function Data_out = delete_after_k(Data_in)
      % Delete the first row (distances of 0)
      Data_in(1,:)=[];
      % Delete all but the first k rows (closest neighbors class indexes)
      Data_in(k+1:end,:) = [];
      Data_out = Data_in;
    end

    function nw = cut_in_half(w)
      if(size(w,1) > 1) 
        nw = w((size(w,1)/2)+1:end);
      end
    end

end


      % Half a Gaussian
      %Im2 = mat2gray(index);
      %imshow(index)
      %predlabels = index;


% random notes: 3d scatter plot?
%{

% traindata = Matrix(ntrain x ndim) --> Each row is one picture, each col
% is one feature.
% trainlabels = Vector (ntrain x 1)
% testdata = Matrix (ntest x ndim)
% k = number of neighbors
% f = distance function to use (ex: 'sqeuclidean')
% D = Matrix (ntrain x ntest) OPTIONAL
% D is an ntrain × ntest matrix of precomputed pairwise distances (optional in the function
% input and your inside code should handle the case where it does not exist, e.g. by calling
% f); and 
 
 %}