
m = 5; %nombre d'individus
n = 10; %nombre d'attributs
p = 0.1; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-3;
delta0=1e-2;

% % recherche du bon r
r = 0.1:0.1:10;
nr = length(r);
iter = zeros(nr,1);
for i=1:nr
    [x,~,flag,iter(i)] = lasso(A,b,x0,z0,u0,lambda,r(i),maxiter,delta,delta0);
end
plot(r,iter);