%% set up data and parameters
clear; close all;

% change the name to gisette if you test gisette data
% load spamData;
load gisette;

lams = [1e-6];
%
% lam1 = 100;
% lam2 = 100;

% make large maxit if needed
maxit = 2000;
tol = 1e-2;

fid = fopen('res_gd.txt', 'a+');


for lam = lams
    
    lam1 = lam;
    lam2 = lam;
    
    %% call the solver LR_gd on the training data
    
    t0 = tic;
    [w_gd, b_gd, hist_obj_gd] = LR_gd(Xtrain,ytrain,lam1,lam2,maxit,tol);
    % time_gd saves the running time for LR_gd
    time_gd = toc(t0);
    

    
    %% plot the objective values of LR_gd and LR_Newton
    fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);
    plot(hist_obj_gd);
    legend('gradient descent');

    %% do classification on testing data
    
    
    N_test = length(ytest);
    y_pred_gd = sign( Xtest*w_gd + b_gd );
    accuracy_pred_gd = sum(y_pred_gd==ytest)/N_test;
    fprintf(fid, '------------------------\n');
    fprintf(fid, '------------------------\n');
    
    fprintf(fid, 'lam1 : %.6f\t lam1 : %.6f\n', lam1, lam2);
    
    fprintf(fid,"\t Gradient Descent:\n");
    fprintf(fid,"\t\tTotol iteration: %d\n", length(hist_obj_gd));
    fprintf(fid,"\t\tTotal run-time: %.4f\n", time_gd);
    fprintf(fid,"\t\tClassification accuracy for test data: %.2f\n\n", accuracy_pred_gd);
    

end

fclose(fid);

