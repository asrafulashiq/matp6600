%% set up data and parameters
clear; close all;

% change the name to gisette if you test gisette data
% load spamData;
load gisette.mat;

lams = 1;%[0.00001, 0.001, 1, 10, 100, 1000];
%
% lam1 = 100;
% lam2 = 100;

% make large maxit if needed
maxit = 100;
tol = 1e-2;



for lam = lams
    
    lam1 = lam;
    lam2 = lam;
    
    %% call the solver LR_gd on the training data
    
    t0 = tic;
    [w_gd, b_gd, hist_obj_gd] = LR_gd(Xtrain,ytrain,lam1,lam2,maxit,tol);
    % time_gd saves the running time for LR_gd
    time_gd = toc(t0);
    

    
    N_test = length(ytest);
    y_pred_gd = sign( Xtest*w_gd + b_gd );
    accuracy_pred_gd = sum(y_pred_gd==ytest)/N_test;
    
    fprintf('GD :lam = %g, score = %g\n', lam, accuracy_pred_gd);
    
    %% call the solver LR_Newton on the training data
        
    
    t0 = tic;
    [w_nt, b_nt, hist_obj_nt] = LR_Newton(Xtrain,ytrain,lam1,lam2,maxit,tol);
    % time_gd saves the running time for LR_gd
    time_gd = toc(t0);
    
    N_test = length(ytest);
    y_pred_nt = sign( Xtest*w_nt + b_nt );
    accuracy_pred_nt = sum(y_pred_nt==ytest)/N_test;
    
    fprintf('NM : lam = %g, score = %g\n', lam, accuracy_pred_nt);

     
    

end


