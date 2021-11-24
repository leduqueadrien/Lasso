
# %% Import
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
from time import time
from lasso import *

# %% Set parameters

m = 1000
n = 10000
p = 100/m/n

r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4

maxiter_temps = 10

# %% Resolution de Ax = b

executions_time = np.zeros(maxiter_temps)
execution_iter =np.zeros(maxiter_temps)

for i in range(maxiter_temps):
	parametres = initialisation(n, m, p)
	start_time = time()
	xk, iter, historique_cout = lasso(*parametres, r, maxiter, epsilone1, epsilone2)
	executions_time[i] = time() - start_time
	execution_iter[i] = iter
    

# %% Affichage

print("temps moyen d'execution : {0}".format(executions_time.mean()))
