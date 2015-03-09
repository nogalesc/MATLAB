% HW2

%% Problem 2.8
clc
clear
load('cars.mat')
slow_rmse = loocvreg_slow(xTr,yTr)
fast_rmse = loocvreg_fast(xTr,yTr)
%%
% 
%  PREFORMATTED
%  TEXT
% 
% load('SenatorVoting.mat')