
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import matplotlib.pyplot as plt

def Seuillage(V, k):
    return np.sign(V) * np.maximum(0, np.abs(V)-k)

def lasso(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2):

    m, n = A.shape
    I = np.eye(n)

    R = lg.cholesky(np.dot(A.T, A) + r*I)


    iter = 0
    fin = False

    while not(fin) :
        iter += 1
        tmp = A.dot(xk_1) - b
        historique_cout.append(np.dot(tmp, tmp.T) + lam*lg.norm(xk_1, ord=1))
        
        tmp = lg.solve(R, np.dot(A.T,b) + r*(zk_1 - uk_1) )
        xk = lg.solve(R.T, tmp)

        zk = Seuillage(xk + uk_1, lam/r)
        e = xk - zk
        uk = uk_1 + e
        
        cl = np.dot(xk - xk_1, xk - xk_1) + np.dot(zk - zk_1, zk - zk_1) + np.dot(uk - uk_1, uk - uk_1)
        cl = cl / np.dot(xk, xk) / np.dot(zk, zk) / np.dot(uk, uk)
        if cl < epsilone1 and e.dot(e) < epsilone2 :
            fin = True
        
        if iter > maxiter:
            fin = True
        
        xk_1 = xk
        zk_1 = zk
        uk_1 = uk

    return xk, iter



m = 5
n = 10
p = 10/m/n

x0 = sp.rand(n, 1, density=p)
A = np.random.normal(size=(m,n))
A = A * sp.diags( 1 / np.sqrt(np.power(A, 2).sum(axis=0)).T, offsets=0, shape=(n,n) )
b = A * x0 + np.sqrt(0.001) * np.random.normal(size=(m, 1))
lambda_max = lg.norm(A.T.dot(b), ord=np.inf)
lam = 0.1*lambda_max

historique_cout = []

x0 = np.zeros(n)
z0 = np.zeros(n)
u0 = np.zeros(n)
r = 1
maxiter = 1000
epsilone1 = 1e-6
epsilone2 = 1e-4
b = b[:,0]

xk, iter = lasso(A, b, x0, z0, u0, lam, r, maxiter, epsilone1, epsilone2)

plt.plot(historique_cout, label="cout")
plt.show()
