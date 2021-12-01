function [A, b, x0, u0, z0, lambda] = init(m,n,p)
    x1 = sprand(n,1,p);
    A = randn(m,n);
    A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);
    b = A*x1 + sqrt(0.001)*randn(m,1);
    x0 = zeros(n,1);
    z0 = zeros(n,1);
    u0 = zeros(n,1);
    lambda_max = norm(A'*b,'inf');
    lambda = 0.1*lambda_max;