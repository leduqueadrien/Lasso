
% % evolution des iterations en fonction des tailles

ns = 100:100:1000;

iters = zeros(length(ns), 1);

nb_iter = 5;

for i=1:length(ns)
	n = ns(i)
	m = n/2;
	p = 0.01;
	iter_n = zeros(nb_iter,1);
	for j=1:nb_iter

		[A,b,x0,z0,u0,lambda]=init(m,n,p);

		[x,~,~,iter_n(j)] = lasso(A,b,x0,z0,u0,lambda,r(i),maxiter,delta,delta0);
	end
	iters(i) = mean(iter_n);
end
ns
iters
semilogy(ns, iters)