
m = 1000; %nombre d'individus
n = 10000; %nombre d'attributs
p = 0.01; %densite de la matrice A

x = sprand(n,1,p);
AS = sprandn(m,n, p);
AS = AS*spdiags(1./sqrt(sum(AS.^2))',0,n,n);
bD = AS*x + sqrt(0.001)*randn(m,1);

AD = full(AS);
bD = full(bD);
bS = sparse(bD);

csvwrite('DataSave/A.csv', AD);
csvwrite('DataSave/b.csv', bD);

RS = chol(AS' * AS + eye(n));
RD = full(RS);


tic
tmpD = AD' * bD;
tmpD = RD' \ tmpD;
xD = RD \ tmpD;
timeD = toc;
fprintf("temps dense %f sec \n", timeD)


tic
tmpS = AS' * bS;
tmpS = RS' \ tmpS;
xS = RS \ tmpS;
timeS = toc;
fprintf("temps sparse %f sec \n", timeS)

