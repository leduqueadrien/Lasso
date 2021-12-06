
m = 5; %nombre d'individus
n = 10; %nombre d'attributs
p = 0.1; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-3;
delta0=1e-2;

r=1;
tic
[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
toc
plot(1:iter, h)


