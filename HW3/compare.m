alphas10 = load('dual_SVM_result_10am.mat'); 
alphas9 = load('dual_SVM_result_9am.mat')

tf = isequal(alphas10.alphas,alphas9.alphas)