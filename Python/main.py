
# %% Import
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import matplotlib.pyplot as plt
from lasso import *

# %% Set parameters

m = 50
n = 100
p = 0.1

r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4

# %% Resolution de Ax = b
A, b, x0, z0, u0, lam = initialisation(m, n, p)
xk, iter, historique_cout = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2)

# %% Affichage

plt.plot(historique_cout, label="cout")
plt.show()

