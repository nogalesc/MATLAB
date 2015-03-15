    function [AA,BB] = randomly_permute_both(A, B)
    % A is a nxm matrix
    % B is a nx1 matrix
    % C is a combo of both
        % Combine
        szA = size(A);
        C = horzcat(A,B);
        % Permunte rows randomly
        C(randperm(size(C,1)),:);
        shuffledC = C(randperm(size(C,1)),:);
        % Uncombine
        AA = shuffledC(1:szA(1),1:szA(2));
        % Delete all but the last szB(2) columns
        shuffledC(:,1:szA(2)) = [];
        BB = shuffledC;
        
    end