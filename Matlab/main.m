m = 50; %nombre d'individus
n = 100; %nombre d'attributs
p = 100/m/n; %densite de la matrice A

x1 = sprand(n,1,p);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);
b = A*x1 + sqrt(0.001)*randn(m,1);

lambda_max = norm(A'*b,'inf');
lambda = 0.1*lambda_max;

x0 = zeros(n,1);
z0 = zeros(n,1);
u0 = zeros(n,1);
r=1;
maxiter=500;
delta=1e-6;
delta0=1e-4;

[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);

plot(1:iter, h)

