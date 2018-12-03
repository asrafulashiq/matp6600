function [w,b,hist_obj] = LR_sgd(X,y,lam1,lam2,maxit,tol)

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

%% initialization
%
% other initial iterate can also be used
w = zeros(n,1);
b = 0;
i = 1;

% grad_w: the partial gradient of w
% grad_b: the partial gradient of b

% write your own function eval_grad to evaluate grad_w and grad_b
[grad_w, grad_b] = eval_grad(w,b,i);

% write your own function eval_obj(w,b)
obj = eval_obj(w,b,i);
hist_obj = obj;
iter = 0; 

%% main iterations
while iter < maxit && ...
        norm(grad_w) + norm(grad_b) >= tol*max(1, norm(w)+norm(b))
    
    iter = iter + 1;  
    i = rem(iter,N) + 1;
    
    % write your own code to choose the step size alpha
    % if backtracking is used, another while-loop should be inserted here
%     alp = 0.1; beta = 0.5;
%     t = 1;
%     f0 = eval_obj(w, b);
%     new_f = eval_obj(w-t*grad_w, b-t*grad_b);
%     
%     while new_f > f0 - alp * t * (norm(grad_w)^2+norm(grad_b)^2)
%        t = beta * t; 
%        new_f = eval_obj(w-t*grad_w, b-t*grad_b);
%     end
%     
%     alpha = t;
%     
    alpha = 0.001;
    
    % update w and b
    
    w = w - alpha*grad_w;
    
    b = b - alpha*grad_b;
    
    % evaluate the gradient for next iteration
    
    [grad_w, grad_b] = eval_grad(w,b,i);
    
    % write your own function eval_obj(w,b)
    obj = eval_obj(w,b,i);
    
    % save the objective value
    hist_obj = [hist_obj; obj];
    
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