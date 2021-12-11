
m = 500; %nombre d'individus
n = 1000; %nombre d'attributs
p = 0.01; %densite de la matrice A

[A,b,x0,z0,u0,lambda]=init(m,n,p);
maxiter=1000;
delta=1e-6;
delta0=1e-4;

r=1;
tic
[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
iter
toc
plot(1:iter, h)
title("Evolution du la fonction de cout au cours de la resolution (m=500, n=1000, r=1, p=1%)")
xlabel('iteration')
ylabel("fonction cout")

print("evolutionFonctionCout.pdf")
