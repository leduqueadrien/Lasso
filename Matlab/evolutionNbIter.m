
% % evolution des iterations en fonction des tailles
maxiter=1000;
delta=1e-6;
delta0=1e-4;
p = 0.01;
r=1

ns = 100:100:5000;

iters = zeros(length(ns), 1);

nb_iter = 5;

for i=1:length(ns)
	n = ns(i)
	m = n/2;
	iter_n = zeros(nb_iter,1);
	for j=1:nb_iter

		[A,b,x0,z0,u0,lambda]=init(m,n,p);

		[x,~,~,iter_n(j)] = lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
	end
	iters(i) = mean(iter_n);
end
ns
iters
semilogy(ns, iters)
title("Evolution du nombre d'iteration en fonction de n (m=n/2, r=1, p=1%)")
xlabel('n')
ylabel("nombre d'iteration")
