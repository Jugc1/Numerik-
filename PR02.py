import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import io
import os
from scipy import sparse


def veclap(n,m):
    Dn=Ableitung(n)
    Dm=Ableitung(m)
    In=  sparse.identity(n)
    Im = sparse.identity(m)
    lap= (sparse.kron(Im,Dn)+sparse.kron(Dm,In))
    return lap
def Ableitung(n):
    D=sparse.eye(n)  *-2
    O=sparse.eye(n,k=1)
    U = sparse.eye(n, k=-1)
    return O+U+D

def diffgl(F,G):
    n,m= np.shape(F)[0],    np.shape(F)[1]
    lapF=veclap(n,m).tocsr().dot(F.ravel("F"))
    return lapF
"""
if __name__ == "__main__":
    K=io.imread(filename)
    BK,GK,RK = K[:, :, 0], K[:, :, 1], K[:, :, 2]
    L=io.imread(filename)
    BL,GL,RL = L[:, :, 0], L[:, :, 1],L[:, :, 2]
    X=io.imread("bear.jpg")
"""