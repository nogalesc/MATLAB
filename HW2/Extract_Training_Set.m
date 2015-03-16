function [X,Y] = Extract_Training_Set(n,XX,YY )
    % How many unique classes do I need? 2
    yu=unique(YY);  % Return the values of the labels {0,1}
    nc=length(yu);  % number of classes = 2
    mode = 0;
    if mod(n,2) == 0
      %number is even
      disp('even');
      half = n/2;
      mode = 0;
    else
      %number is odd
      disp('odd');
      half = floor(n/2);
      mode = 1;
    end 
    % Dummy 
    X = ones(1,size(XX,2));
    Y = ones(1,1);
    for i=1:nc
        % All samples for each class
        xi=XX((YY==yu(i)),:); 
        % All samples for each class
        yi=YY((YY==yu(i)),:); 
        % If it was odd, add an extra one at the last iteration
%         if(mode == 1 && i==nc)
%            half = half + 1;
%         end
        % Extract the first n/2 regardless
%         half
%         whos xi
%         whos X
        X = vertcat(X,xi(1:half,:));
        Y = vertcat(Y,yi(1:half,:));
    end
    % Get rid of first dummy value we had
    X(1,:) = [];
    Y(1,:) = [];
end