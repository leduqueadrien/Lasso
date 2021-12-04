
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import scipy.stats as st
from scipy.sparse import linalg as splg


def Seuillage(V, k):
    return np.sign(V) * np.maximum(0, np.abs(V)-k)

def lasso(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2):

    m, n = A.shape
    I = sp.eye(n)
    historique_f_cout = []
    iter = 0
    fin = False
    
    R = lg.cholesky((np.dot(A.T, A) + I).todense()).T
    R = sp.csr_matrix(R)

    while not(fin) :
        iter += 1
        tmp = A.dot(xk_1) - b
        historique_f_cout.append(np.dot(tmp, tmp.T) + lam*lg.norm(xk_1, ord=1))
        
        tmp = splg.spsolve(R.T, A.T.dot(b) + r*(zk_1 - uk_1) )
        xk = splg.spsolve(R, tmp)
        
        zk = Seuillage(xk + uk_1, lam/r)
        e = xk - zk
        uk = uk_1 + e
        
        cl = np.dot(xk - xk_1, xk - xk_1) + np.dot(zk - zk_1, zk - zk_1) + np.dot(uk - uk_1, uk - uk_1)
        cl = cl / (np.dot(xk, xk) + np.dot(zk, zk) + np.dot(uk, uk))
        
        if np.sqrt(cl) < epsilone1 and lg.norm(e, 2) < epsilone2 :
            fin = True
        if iter > maxiter:
            fin = True
        
        xk_1 = xk
        zk_1 = zk
        uk_1 = uk

    return xk, iter, historique_f_cout

def initialisation(m, n, p):
    normal = st.norm().rvs
    x0 = sp.random(n,1, density=p, data_rvs=normal)
    A = sp.random(m, n, density=p, data_rvs=normal)
    A = A * sp.diags( np.array(1 / np.sqrt(A.power(2).sum(axis=0)))[0,:], offsets=0)
    b = A * x0 + np.sqrt(0.001) * np.random.normal(size=(m, 1))
    lambda_max = lg.norm(A.T.dot(b), ord=np.inf)
    lam = 0.1*lambda_max

    x0 = np.zeros(n)
    z0 = np.zeros(n)
    u0 = np.zeros(n)
    b = np.array(b)[:,0]
    
    return A, b, x0, z0, u0, lam
