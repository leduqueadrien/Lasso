
from time import time
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import scipy.stats as st
from scipy.sparse import linalg as splg
import scipy.linalg as sl


m = 1000
n = 10000

AD = np.loadtxt("DataSave/A.csv", delimiter=',')
AS = sp.csr_matrix(AD)

bD = np.loadtxt("DataSave/b.csv", delimiter=',')
bS = sp.csr_matrix(bD)

# Upper matrice
RD = lg.cholesky(np.dot(AD.T, AD) + np.eye(n)).T
RS = sp.csr_matrix(RD)

start_time_D = time()
tmpD = AD.T.dot(bD)
tmpD = lg.solve(RD.T, tmpD)
xD = lg.solve(RD,  tmpD)
print(f"temps dense : {time() - start_time_D}")

start_time_D = time()
tmpD = AD.T.dot(bD)
tmpD = sl.solve_triangular(RD.T, tmpD, lower=True)
xD = sl.solve_triangular(RD,  tmpD, lower=False)
print(f"temps dense triangulaire : {time() - start_time_D}")


start_time_S = time()
tmpS = AS.T.dot(bS.T)
tmpS = splg.spsolve(RS.T, tmpS)
xS = splg.spsolve(RS,  tmpS)
print(f"temps sparse : {time() - start_time_S}")


start_time_S = time()
tmpD = AS.T.dot(bD.T)
tmpD = splg.spsolve_triangular(RS.T, tmpD, lower=True)
xS = splg.spsolve_triangular(RS,  tmpD, lower=False)
print(f"temps sparse triangulaire: {time() - start_time_S}")

