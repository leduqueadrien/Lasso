
# Import
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
from time import time
from lasso import *

# Set parameters
m = 50
n = 100
p = 0.1
r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4
maxiter_temps = 50

# Resolution de Ax = b

executions_time = np.zeros(maxiter_temps)
execution_iter = np.zeros(maxiter_temps)
execution_perf = np.zeros(maxiter_temps)

for i in range(maxiter_temps):
	parametres = initialisation(m, n, p)
	start_time = time()
	xk, iter, historique_cout, flags = lasso(*parametres, r, maxiter, epsilone1, epsilone2)
	executions_time[i] = time() - start_time
	execution_iter[i] = iter
	execution_perf[i] = historique_cout[iter-1]
    

# Affichage

print("performance moyenne : {0}".format(execution_perf.mean()))
print("temps moyen d'execution : {0}".format(executions_time.mean()))
print("nombre moyen d'iteration : {0}".format(execution_iter.mean()))
