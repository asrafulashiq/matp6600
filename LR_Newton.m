function [w,b,hist_obj] = LR_Newton(X,y,lam1,lam2,maxit,tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Newton's method for logistic regression problem:
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

% grad_w: the partial gradient of w
% grad_b: the partial gradient of b

% write your own function eval_grad to evaluate grad_w and grad_b
[grad_w, grad_b] = eval_grad(w,b);
[H_w, H_b] = eval_H(w, b);

% write your own function eval_obj(w,b)
obj = eval_obj(w, b);
hist_obj = obj;
iter = 0;

%% main iterations
while iter < maxit && ...
        norm(grad_w) + norm(grad_b) >= tol*max(1, norm(w)+norm(b))
    
    iter = iter + 1;
    del_w = -(H_w\grad_w);
    del_b = -(H_b\grad_b);
    % write your own code to choose the step size alpha
    % if backtracking is used, another while-loop should be inserted here
%     alp = 0.1; beta = 0.5;
%     t = 1;
%     f0 = eval_obj(w, b);
%     del = (grad_w' * del_w + grad_b * del_b );
%     new_f = eval_obj(w+t*del_w, b+t*del_b);
%     
%     while new_f > f0 + alp * t * del
%        t = beta * t; 
%        new_f = eval_obj(w+t*del_w, b+t*del_b);
%     end
%     
%     alpha = t;
    
    alpha = 1;
        
    % update w and b
    
%     w = w - alpha * (inv(H_w) * grad_w);
%     b = b - alpha * (inv(H_b) * grad_b);
    w = w + alpha * del_w;
    b = b + alpha * del_b;
    
    % evaluate the gradient for next iteration
    
    [grad_w, grad_b] = eval_grad(w,b);
    [H_w, H_b] = eval_H(w, b);
    
    % write your own function eval_obj(w,b)
    obj = eval_obj(w, b);
    
    % save the objective value
    hist_obj = [hist_obj; obj];

    
end % of main iteration


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [grad_w, grad_b] = eval_grad(w,b)
        
        e_theta = exp(-(X*w+b).*y);
        alph = e_theta ./ (1+e_theta) .* (-y);
        grad_w = 1/N * X' * alph + lam1 * w;
        grad_b = 1/N * sum(alph) + lam2 * b;
        
        % complete this function
    end % of function eval_grad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [H_w, H_b] = eval_H(w, b)
        e_theta = exp(-(X*w+b).*y);
        A = 1/N * e_theta ./ (1+e_theta).^2;     
        
        D = diag(A);
        H_w = X'*D*X + lam1*eye(n);
        H_b = sum(A) + lam2;

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function obj = eval_obj(w, b)
        e_theta = exp(-(X*w+b).*y);        
        obj = 1/N * sum(log(1+e_theta)) + lam1/2 * norm(w)^2 + lam2/2 * b^2;  
        % complete this function
    end % of function eval_obj

end