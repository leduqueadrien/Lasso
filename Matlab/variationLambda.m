
m = 500; %nombre d'individus
n = 1000; %nombre d'attributs
p = 0.01; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-6;
delta0=1e-4;
r=1;

% variation du lambda
lambda_max = norm(A'*b,'inf');
lambda = lambda_max * (0.01:0.1:1);
nl = length(lambda);
vect = zeros(nl,1);
for i=1:nl
    [x,~,flag,~] = lasso(A,b,x0,z0,u0,lambda(i),r,maxiter,delta,delta0);
    vect(i) = sum(abs(x)<1e-6);
end
plot(lambda,vect)
title("Evolution du nombre de zeros dans la solution en fonction de lambda (m=500, n=1000, r=1, p=1%)")
xlabel('lambda')
ylabel("nombre de zeros dans la solution")
