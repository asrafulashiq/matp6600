%% set up data and parameters
clear; close all;

% change the name to gisette if you test gisette data
load spamData;
% load gisette.mat;


lam = 0.001;

maxit = 30000;
tols = 1e-4;


for tol = tols
    
    lam1 = lam;
    lam2 = lam;
    
    %% call the solver LR_gd on the training data
    
    t0 = tic;
    [w_gd, b_gd, hist_obj_gd] = LR_sgd_back(Xtrain,ytrain,lam1,lam2,maxit,tol);
    % time_gd saves the running time for LR_gd
    time_gd = toc(t0);
    
    %% plot the objective values of LR_gd and LR_Newton
    fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
    plot(hist_obj_gd);
    xlabel('Iteration');
    ylabel('Loss function');
    legend(sprintf('stochastic gradient descent'));

    %% do classification on testing data
    
    
    N_test = length(ytest);
    y_pred_gd = sign( Xtest*w_gd + b_gd );
    accuracy_pred_gd = sum(y_pred_gd==ytest)/N_test;
    
    fprintf('%g : %g\t%g\t%g\n', tol, length(hist_obj_gd), time_gd, accuracy_pred_gd);

end


