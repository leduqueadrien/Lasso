
from lasso import *
from time import time

m = 1000
n = 10000
p = 0.01
r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4

A, b, x0, z0, u0, lam = initialisation(m, n, p, "../ComparaisonMatlabPython/DataSave/A.csv", "../ComparaisonMatlabPython/DataSave/b.csv")

start_time = time()
xk, iter, historique_cout, flags = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2, compute_dense_choleski=True)

print(f"temps d'execution : {time()-start_time} sec")
print(f"nombre d'iteration : {iter}")

