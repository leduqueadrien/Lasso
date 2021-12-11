
# Import
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import matplotlib.pyplot as plt
from lasso import *
from time import time
# Set parameters

m = 1000
n = 10000
p = 0.1
r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4

# Resolution de Ax = b
A, b, x0, z0, u0, lam = initialisation(m, n, p)
start_time = time()
xk, iter, historique_cout, flags = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2)
print(f"temps d'execution : {time() - start_time}")
# Affichage

plt.plot(historique_cout, label="cout")
# plt.yscale("log")
plt.show()

