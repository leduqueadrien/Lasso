
m = 5; %nombre d'individus
n = 10; %nombre d'attributs
p = 0.1; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-3;
delta0=1e-2;

% variation du lambda
r=1;
lambda_max = norm(A'*b,'inf');
lambda = lambda_max * (0.01:0.1:1);
nl = length(lambda);
vect = zeros(nl,1);
for i=1:nl
    [x,~,flag,~] = lasso(A,b,x0,z0,u0,lambda(i),r,maxiter,delta,delta0);
    vect(i) = sum(abs(x)<1e-6);
end
plot(lambda,vect)