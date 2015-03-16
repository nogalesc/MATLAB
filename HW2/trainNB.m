function [ confidence , pv] = trainNB(X,Y,u,v)
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
        if(size(xi,1) ==1)
            mu(i,:)=mean(xi,1);         % 2x543 matrix
            sigma(i,:) = ones(1,nf);
        else
            mu(i,:)=mean(xi,1);         % 2x543 matrix
            sigma(i,:)=std(xi,1);       % 2x543 matrix
        end
    end
    whos mu
    whos sigma
    % probability for test set
    for j=1:ns
        % For each candidate, and for each class, returns the normal
        % probability
        temp_a = ones(nc,1);
        temp_b = u(j,:);
%         nc
%         j
%         whos temp_a
%         whos temp_b
%         whos mu
%         whos sigma
        fu=normcdf(temp_a*temp_b,mu,sigma);
        % multiply all probabilities from each feature (row-wise)
        P(j,:)=fy.*prod(fu,2)';     % 16x2 matrix
    end
    % get predicted output for each sample in the test set
    [~,id]=max(P,[],2);           % index of maximum probability
    for i=1:length(id)
        pv(i,1)=yu(id(i));        % convert from index to {0,1}
    end
    confidence=sum(pv==v)/length(pv);
%     fprintf('\nNB Test Set Accuracy: %f\n',confidence);   
end

