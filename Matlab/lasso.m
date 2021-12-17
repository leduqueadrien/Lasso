function [x,h,flag,iter] = lasso(A, b, x0, z0, u0, lambda, r, maxiter, eps1, eps2)
    %%% Probleme du LASSO avec ADMM %%%
    % flags : 
    % 0 : OK
    % 1 : nombre max d'iter atteint
    % 2 : A'*A + r*eye(n) pas défini positive, faut mettre un plus grand r
    [~,n]=size(A);
    u = u0;
    x = x0;
    z = z0;
    iter = 0;
    flag = 0;
    done = false;
    h = zeros(maxiter,1);
    [M, P] = chol(A'*A + r*eye(n));
    if (P~=0)
        flag = 2;
        done = true;
    end
    
    while iter < maxiter && not(done)
        x_anc = x;
        z_anc = z;
        u_anc = u;
        temp = A' * b + r*(z-u);
        x = M \ ( M' \ temp);
        z = sto(x+u, lambda/r);
        e = x - z;
        u = u + e;
        
        c1 = ((z-z_anc)'*(z-z_anc) + (x-x_anc)'*(x-x_anc) + (u-u_anc)'*(u-u_anc)) / (z'*z + x'*x + u'*u);
        
        iter = iter + 1;
        h(iter)=0.5 * norm(A*x-b)^2 + lambda * norm(x,1);
        done = (sqrt(c1) < eps1) && (norm(e) < eps2);
    end
    
    h = h(1:iter);
    
    if iter == maxiter
        flag = 1;
    end
end