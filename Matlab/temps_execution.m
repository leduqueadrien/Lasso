
m = 50; %nombre d'individus
n = 100; %nombre d'attributs
p = 0.1; %densite de la matrice A
r=1;
maxiter=1000;
delta=1e-6;
delta0=1e-4;

maxiter_temps = 50;
executions_time = zeros(maxiter_temps,1);
execution_iter = zeros(maxiter_temps,1);
execution_perf = zeros(maxiter_temps,1);

for i=1:maxiter_temps
	[A,b,x0,z0,u0,lambda]=init(m,n,p);
	tic
	[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
	executions_time(i) = toc;
	execution_iter(i) = iter;
	execution_perf(i) = h(iter);
end

disp("performance moyenne : ")
mean(execution_perf)
disp("temps moyen d'execution : ")
mean(executions_time)
disp("nombre moyen d'iteration : ")
mean(execution_iter)