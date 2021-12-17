
% % evolution des iterations en fonction des tailles
maxiter=1000;
eps1=1e-6;
eps2=1e-4;
p = 0.01;
r=1;

ns = [100 500 1000];

moyenne = zeros(length(ns), 1);
variance = zeros(length(ns), 1);

nb_iter = 100;

for i=1:length(ns)
	n = ns(i);
	m = n/2;
	iter_n = zeros(nb_iter,1);
	for j=1:nb_iter

		[A,b,x0,z0,u0,lambda]=init(m,n,p);

		[x,~,~,iter_n(j)] = lasso(A,b,x0,z0,u0,lambda,r,maxiter,eps1,eps2);
	end
	moyenne(i) = mean(iter_n);
	variance(i) = std(iter_n);
end

fprintf("Nombre d'iteration moyen pour n=100, m=50, r=1 et p=0.01 :\n")
fprintf("%f\n\n", moyenne(1))

fprintf("Nombre d'iteration moyen pour n=500, m=250, r=1 et p=0.01 :\n")
fprintf("%f\n\n", moyenne(2))

fprintf("Nombre d'iteration moyen pour n=1000, m=500, r=1 et p=0.01 :\n")
fprintf("%f\n\n", moyenne(3))

fprintf("Ecart type du nombre d'iteration pour n=100, m=50, r=1 et p=0.01 :\n")
fprintf("%f\n\n", variance(1))

fprintf("Ecart type du nombre d'iteration pour n=500, m=250, r=1 et p=0.01 :\n")
fprintf("%f\n\n", variance(2))

fprintf("Ecart type du nombre d'iteration pour n=1000, m=500, r=1 et p=0.01 :\n")
fprintf("%f\n\n", variance(3))


