function [w,b,hist_obj] = LR_sgd_back(X,y,lam1,lam2,maxit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% steepest gradient descent for logistic regression problem:
%
% min_{w,b} 1/N*sum_{i=1}^N log( 1+exp[-yi*(w'*xi+b)] )
%           + .5*lam1*w'*w + .5*lam2*b^2
%
% input:
%       X(i,:) is the i-th data point
%       y(i) is the label
%       lam1, lam2: parameters in the model
%       maxit: maximum number of iterations
%       tol: stopping tolerance
%
% output:
%       w, b: approximation solution of the model
%       hist_obj: objective values at all iterates
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% get size of the data
%
% N is the number of sample points
% n is the dimension of each sample point

[N,n] = size(X);

% shuffle data
rand_ind = randperm(N);
X = X(rand_ind, :);
y = y(rand_ind);

%% initialization
%
% other initial iterate can also be used
w = zeros(n,1);
b = 0;

hist_obj = [];
iter = 0;

%% main iterations
while iter < maxit    
    
    iter = iter + 1;
    i = rem(iter,N) + 1;
    
    % write your own function eval_grad to evaluate grad_w and grad_b
    [grad_w, grad_b] = eval_grad(w,b,i);
    
    % write your own function eval_obj(w,b)
    obj = eval_obj(w,b,i);
    
     alpha = 1e-5;%/sqrt(iter);
    
    % update w and b
    
    w = w - alpha*grad_w;
    
    b = b - alpha*grad_b;
    hist_obj = [hist_obj; obj];
    
%     if norm(grad_w) + norm(grad_b) >= tol*max(1, norm(w)+norm(b))
%        break; 
%     end
    
end % of main iteration



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [grad_w, grad_b] = eval_grad(w,b,i)
        xi = X(i,:)';
        yi = y(i);
        
        e_theta = exp(-(xi'*w+b).*yi);
        alph = e_theta ./ (1+e_theta) .* (-yi);
        grad_w = xi * alph + lam1 * w;
        grad_b = alph + lam2 * b;
        
        % complete this function
    end % of function eval_grad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function obj = eval_obj(w,b,i)
        xi = X(i,:)';
        yi = y(i);
        e_theta = exp(-(xi'*w+b).*yi);
        
        obj = log(1+e_theta) + lam1/2 * norm(w)^2 + lam2/2 * b^2;
        
        % complete this function
    end % of function eval_obj

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end