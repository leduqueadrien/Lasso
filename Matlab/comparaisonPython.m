
m = 1000; %nombre d'individus
n = 10000; %nombre d'attributs
p = 0.01; %densite de la matrice A
r=1;
maxiter=1000;
eps1=1e-6;
eps2=1e-4;


[A,b,x0,z0,u0,lambda]=init(m,n,p);
csvwrite('../ComparaisonMatlabPython/DataSave/A.csv', full(A))
csvwrite('../ComparaisonMatlabPython/DataSave/b.csv', b)
tic
[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,eps1,eps2);
executions_time = toc;

fprintf("temps d'execution %f sec\n", executions_time)
fprintf("nombre d'iterqtion %d", iter)

