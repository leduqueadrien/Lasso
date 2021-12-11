
m = 500; %nombre d'individus
n = 1000; %nombre d'attributs
p = 0.01; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-6;
delta0=1e-4;

% % recherche du bon r
r = 0.05:0.05:10;
nr = length(r);
iter = zeros(nr,1);
for i=1:nr
    [x,~,flag,iter(i)] = lasso(A,b,x0,z0,u0,lambda,r(i),maxiter,delta,delta0);
end
plot(r,iter);
title("Evolution du nombre d'iteration en fonction de r (m=500, n=1000, p=1%)")
xlabel('r')
ylabel("nombre d'iteration")
