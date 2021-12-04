
# Import
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import matplotlib.pyplot as plt
from lasso import *

# Set parameters

m = 5
n = 10
p = 0.1
r = 1
maxiter = 1000
epsilone1 = 1e-3
epsilone2 = 1e-2

# Resolution de Ax = b
A, b, x0, z0, u0, lam = initialisation(m, n, p)
xk, iter, historique_cout, flags = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2)

# Affichage

plt.plot(historique_cout, label="cout")
# plt.yscale("log")
plt.show()

