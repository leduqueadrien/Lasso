m = 50; %nombre d'individus
n = 100; %nombre d'attributs
p = 100/m/n; %densite de la matrice A

x1 = sprand(n,1,p);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);
b = A*x1 + sqrt(0.001)*randn(m,1);

lambda_max = norm(A'*b,'inf');
lambda = 0.1*lambda_max;

x0 = zeros(n,1);
z0 = zeros(n,1);
u0 = zeros(n,1);
maxiter=10000;
delta=1e-6;
delta0=1e-4;
% execution classique
r=1;
tic
[x,h,flag,iter]=lasso(A,b,x0,z0,u0,lambda,r,maxiter,delta,delta0);
toc
plot(1:iter, h)

% % recherche du bon r
% r = 0.1:0.1:10;
% nr = length(r);
% iter = zeros(nr,1);
% for i=1:nr
%     [x,~,flag,iter(i)] = lasso(A,b,x0,z0,u0,lambda,r(i),maxiter,delta,delta0);
% end
% plot(r,iter);

% % evolution des iterations en fonction des tailles

% n = 100:100:10000; %nombre d'attributs
% m = n/2;
% p = 100/m/n; %densite de la matrice A
% 
% x1 = sprand(n,1,p);
% A = randn(m,n);
% A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n);
% b = A*x1 + sqrt(0.001)*randn(m,1);
% 
% lambda_max = norm(A'*b,'inf');
% lambda = 0.1*lambda_max;
