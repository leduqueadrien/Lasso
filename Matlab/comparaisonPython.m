
m = 1000; %nombre d'individus
n = 10000; %nombre d'attributs
p = 0.01; %densite de la matrice A
r=1;
maxiter=1000;
delta=1e-6;
delta0=1e-4;


[A,b,x0,z0,u0,lambda]=init(m,n,p);
csvwrite('../DataSave/A.csv', full(A))
csvwrite('../DataSave/b.csv', b)
tic
[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
executions_time = toc;

fprintf("temis moyen d'execution %f sec", executions_time)

