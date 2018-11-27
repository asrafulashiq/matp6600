%% set up data and parameters
clear; close all;

% change the name to gisette if you test gisette data
% load spamData;
load gisette;
lam1 = 0.001;
lam2 = 0.001;

% make large maxit if needed
maxit = 1000;
tol = 1e-2;

%% call the solver LR_gd on the training data

t0 = tic;
[w_gd, b_gd, hist_obj_gd] = LR_gd(Xtrain,ytrain,lam1,lam2,maxit,tol);
% time_gd saves the running time for LR_gd
time_gd = toc(t0);

%% call the solver LR_Newton on the training data

t0 = tic;
[w_nt, b_nt, hist_obj_nt] = LR_Newton(Xtrain,ytrain,lam1,lam2,maxit,tol);
% time_nt saves the running time for LR_Newton
time_nt = toc(t0);

%% plot the objective values of LR_gd and LR_Newton
fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
plot(hist_obj_gd);
% change the name for gisette data
print(fig,'-dpdf','obj_gd_spamData');

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
plot(hist_obj_nt);
% change the name for gisette data
print(fig,'-dpdf','obj_nt_spamData');

figure(3);
plot(hist_obj_gd);
hold on;
plot(hist_obj_nt);
legend('gradient descent', 'newton');
hold off;


%% do classification on testing data
N_test = length(ytest);
y_pred_gd = sign( Xtest*w_gd + b_gd );
accuracy_pred_gd = sum(y_pred_gd==ytest)/N_test;

y_pred_nt = sign( Xtest*w_nt + b_nt );
accuracy_pred_nt = sum(y_pred_nt==ytest)/N_test;
