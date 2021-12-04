
import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import scipy.stats as st
from scipy.sparse import linalg as splg


def Seuillage(V, k):
    # Fonction de seuillage de V
    # V est un vecteur
    # retourne :
    # V[i] - k  si V[i] > k
    # 0         si abs(V[i]) <= k
    # V[i] + k  si V[i] < -k
    return np.sign(V) * np.maximum(0, np.abs(V)-k)


def sparse_choleski(A):
    # retourne la matrice Superieur de la decomposition de choleski de A
    # L'entree et la sortie sont des matrices creuses
    R = lg.cholesky(A.todense()).T
    return sp.csr_matrix(R)


def lasso(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2):
    if sp.issparse(A):
        return lasso_sparse(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2)
    else:
        return lasso_dense(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2)


def lasso_sparse(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2):
    # Resout le probleme :
    # min 1/2*||Ax-b||_2^2 + lam*||x||_1
    # 
    # Entree :
    # A : matrice creuse du probleme
    # b : vecteur du probleme
    #  xk_1, zk_1, uk_1 : valeurs initials des variables du primal et dual
    # lam : parametre du probleme
    # r : parametre du lagrangien augmente
    # maxiter : nombre maximal d'iteration
    # epsilone1 et epsilone2 : condition de fin
    # 
    # Sortie :
    # xk : valeur trouvee de x a l'optimum
    # iter : nombre d'iteration pour attendre l'optimum
    # historique_f_cout : historique de la valeur de la fonction objectif
    # flags :
    #           flags=0 : arret naturel de l'algorithme
    #           flags=1 : arret par nombre maximal d'iteration atteint
        
    m, n = A.shape
    iter = 0
    fin = False
    historique_f_cout = np.zeros(maxiter)
    flags = 0
    
    # On calcul la decomposition de choleski
    R = sparse_choleski(np.dot(A.T, A) + sp.eye(n))

    while not(fin) :
        
        # Actualisation de x, z et u, les variables du primal et dual
        xk = splg.spsolve(R, splg.spsolve(R.T, A.T.dot(b) + r*(zk_1 - uk_1)) )
        zk = Seuillage(xk + uk_1, lam/r)
        e = xk - zk
        uk = uk_1 + e
        
        # Calcul des condition d'arret
        cl = np.dot(xk - xk_1, xk - xk_1) + np.dot(zk - zk_1, zk - zk_1) + np.dot(uk - uk_1, uk - uk_1)
        cl = cl / (np.dot(xk, xk) + np.dot(zk, zk) + np.dot(uk, uk))
        
        erreur_systeme = A.dot(xk) - b
        
        # Test d'arret
        if np.sqrt(cl) < epsilone1 and lg.norm(e, 2) < epsilone2 :
            fin = True
        if iter > maxiter:
            fin = True
            flags = 1
        
        # Memorisation des valeurs des variables primal et dual
        xk_1 = xk
        zk_1 = zk
        uk_1 = uk
        
        # Calcul de la fonction objectif
        historique_f_cout[iter] = 0.5*np.dot(erreur_systeme, erreur_systeme.T) + lam*lg.norm(xk, ord=1)
        
        # Incrementation du compteur d'iteration
        iter += 1

    return xk, iter, historique_f_cout[:iter], flags


def lasso_dense(A, b, xk_1, zk_1, uk_1, lam, r, maxiter, epsilone1, epsilone2):
    # Resout le probleme :
    # min 1/2*||Ax-b||_2^2 + lam*||x||_1
    # 
    # Entree :
    # A : matrice dense du probleme
    # b : vecteur du probleme
    #  xk_1, zk_1, uk_1 : valeurs initials des variables du primal et dual
    # lam : parametre du probleme
    # r : parametre du lagrangien augmente
    # maxiter : nombre maximal d'iteration
    # epsilone1 et epsilone2 : condition de fin
    # 
    # Sortie :
    # xk : valeur trouvee de x a l'optimum
    # iter : nombre d'iteration pour attendre l'optimum
    # historique_f_cout : historique de la valeur de la fonction objectif
    # flags :
    #           flags=0 : arret naturel de l'algorithme
    #           flags=1 : arret par nombre maximal d'iteration atteint
        
    m, n = A.shape
    iter = 0
    fin = False
    historique_f_cout = np.zeros(maxiter)
    flags = 0
    
    # On calcul la decomposition de choleski
    R = lg.cholesky(np.dot(A.T, A) + sp.eye(n)).T

    while not(fin) :
        
        # Actualisation de x, z et u, les variables du primal et dual
        xk = lg.solve(R, lg.solve(R.T, A.T.dot(b) + r*(zk_1 - uk_1)) )
        zk = Seuillage(xk + uk_1, lam/r)
        e = xk - zk
        uk = uk_1 + e
        
        # Calcul des condition d'arret
        cl = np.dot(xk - xk_1, xk - xk_1) + np.dot(zk - zk_1, zk - zk_1) + np.dot(uk - uk_1, uk - uk_1)
        cl = cl / (np.dot(xk, xk) + np.dot(zk, zk) + np.dot(uk, uk))
        
        erreur_systeme = A.dot(xk) - b
        
        # Test d'arret
        if np.sqrt(cl) < epsilone1 and lg.norm(e, 2) < epsilone2 :
            fin = True
        if iter > maxiter:
            fin = True
            flags = 1
        
        # Memorisation des valeurs des variables primal et dual
        xk_1 = xk
        zk_1 = zk
        uk_1 = uk
        
        # Calcul de la fonction objectif
        historique_f_cout[iter] = 0.5*np.dot(erreur_systeme, erreur_systeme.T) + lam*lg.norm(xk, ord=1)
        
        # Incrementation du compteur d'iteration
        iter += 1

    return xk, iter, historique_f_cout[:iter], flags


def initialisation(m, n, p):
    # creer les parametres du probleme : 
    # min 1/2*||Ax-b||_2^2 + lambda*||x||_1
    # Ou A est une matrice definie positive
    # A matrice sparse
    # A : m x n
    # p : densite de valeurs differentes de zeros dans A
    # 
    # Sortie :
    # A : matrice, parametre du probleme
    # b : vecteur, parametre du probleme
    # x0 : vecteur initial de la variable primal
    # z0 : vecteur initial de la deuxieme variable du primal
    # u0 : vecteur initial de la variable dual
    
    # On genere un x, solution de l'eqution Ax=b
    x = sp.random(n,1, density=p, data_rvs=st.norm().rvs)
    
    # On genere la matrice A : m x n
    A = sp.random(m, n, density=p, data_rvs=st.norm().rvs)
    # On rend A define positif en rendant sa diagonal dominante
    A = A * sp.diags( np.array(1 / np.sqrt(A.power(2).sum(axis=0)))[0,:], offsets=0)
    
    # On genere b second membre de l'equation Ax=b. On y ajoute un faible bruit gaussien
    b = A * x + np.sqrt(0.001) * np.random.normal(size=(m, 1))
    
    # On calcul le parametre du probleme lambda
    lambda_max = lg.norm(A.T.dot(b), ord=np.inf)
    lam = 0.1*lambda_max

    # On genere les valeurs initials des variables primal et dual
    x0 = np.zeros(n)
    z0 = np.zeros(n)
    u0 = np.zeros(n)
    b = np.array(b)[:,0]
    
    return A, b, x0, z0, u0, lam
