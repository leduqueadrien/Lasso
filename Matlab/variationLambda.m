
m = 100; %nombre d'individus
n = 500; %nombre d'attributs
p = 0.01; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-6;
delta0=1e-4;
r=1;

% variation du lambda
lambda_max = norm(A'*b,'inf');
lambda = lambda_max * (0.01:0.001:1);
nl = length(lambda);
vect = zeros(nl,1);
for i=1:nl
    lambda(i)
    [x,~,flag,~] = lasso(A,b,x0,z0,u0,lambda(i),r,maxiter,delta,delta0);
    vect(i) = sum(abs(x)<1e-6)/n;
end
plot(lambda,vect)
title("Evolution du pourcentage de zeros dans la solution en fonction de lambda (m=100, n=500, r=1, p=1%)")
xlabel('lambda')
ylabel("pourcentage de zeros dans la solution")
print("variationLambda.pdf")

