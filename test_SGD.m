%% set up data and parameters
clear; close all;

% change the name to gisette if you test gisette data
load spamData;
% load gisette.mat;

% lams = [1e-6];

lam = 0.001;
% lam1 = 100;
% lam2 = 100;

% make large maxit if needed
maxit = 100000;
% tols = [1e-2, 1e-4, 1e-6];
tols = 1e-4;
% fid = fopen('res_sgd.txt', 'a+');


for tol = tols
    
    lam1 = lam;
    lam2 = lam;
    
    %% call the solver LR_gd on the training data
    
    t0 = tic;
    [w_gd, b_gd, hist_obj_gd] = LR_sgd_back(Xtrain,ytrain,lam1,lam2,maxit,tol);
    % time_gd saves the running time for LR_gd
    time_gd = toc(t0);
    
    %% get optimal solution
%     tol = 1e-10;
%     [w_opt, b_opt, hist_obj_opt] = LR_nt1(Xtrain,ytrain,lam1,lam2,maxit,tol);
%     obj_opt = min(hist_obj_opt);
    obj_opt = 4.2996e-09;
    
    %% plot the objective values of LR_gd and LR_Newton
    fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
%     semilogy(hist_obj_gd-obj_opt, 'LineWidth', 1);
    plot(hist_obj_gd);
    xlabel('Iteration');
    ylabel('Loss function');
    legend(sprintf('stochastic gradient descent'));

    %% do classification on testing data
    
    
    N_test = length(ytest);
    y_pred_gd = sign( Xtest*w_gd + b_gd );
    accuracy_pred_gd = sum(y_pred_gd==ytest)/N_test;
    
    fprintf('%g : %g\t%g\t%g\n', tol, length(hist_obj_gd), time_gd, accuracy_pred_gd);
    
%     fprintf(fid, '------------------------\n');
%     fprintf(fid, '------------------------\n');
%     
%     fprintf(fid, 'tol : %.6f\t lam1 : %.6f\t lam1 : %.6f\n', tol, lam1, lam2);
%     
%     fprintf(fid,"\tStochastic Gradient Descent:\n");
%     fprintf(fid,"\t\tTotol iteration: %d\n", length(hist_obj_gd));
%     fprintf(fid,"\t\tTotal run-time: %.4f\n", time_gd);
%     fprintf(fid,"\t\tClassification accuracy for test data: %.2f\n\n", accuracy_pred_gd);
%     

end

% fclose(fid);

