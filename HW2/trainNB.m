function [ theta ] = trainNB(X,y)
% for each value of that feature c of K possible values: do
% for each value j of Y in {0,1} do
    k = 2; % Number of possible features
    
    nestedfx
    theta = 0;
    
   function countY
      disp('This is the nested function')
   end
   % Num of (Xi given Xi = c, Y = j) + alpha
   % Num of (Y = j) + alpha*K
end

