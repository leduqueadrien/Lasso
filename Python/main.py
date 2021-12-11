
# Import
import matplotlib.pyplot as plt
from lasso import *
from time import time
# Set parameters

m = 100
n = 1000
p = 0.01
r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4

# Resolution de Ax = b
file_A = "A.csv"
file_b = "b.csv"
A, b, x0, z0, u0, lam = initialisation(m, n, p, file_A, file_b)
start_time = time()
xk, iter, historique_cout, flags = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2)
print(f"temps d'execution : {time() - start_time}")

# Affichage
plt.plot(historique_cout, label="cout")
# plt.yscale("log")
plt.show()

