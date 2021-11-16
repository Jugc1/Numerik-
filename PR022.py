import matplotlib.pyplot as plt
from math import floor
from math import ceil
import math
import matplotlib.image as mpimg
import numpy as np
from numpy.linalg import norm
import skimage
from skimage import io
import os
from scipy import sparse
from scipy.sparse.linalg import cg


def veclap(n,m):
    Dn=Ableitung(n)
    Dm=Ableitung(m)
    In=  sparse.identity(n)
    Im = sparse.identity(m)
    lap= (sparse.kron(Im,Dn)+sparse.kron(Dm,In))
    return lap

def div(V,n,m):
    DNR=  sparse.eye(n, k=-1)*-1+ sparse.identity(n)
    DMR=  sparse.eye(m, k=-1)*-1+ sparse.identity(m)
    A=sparse.csr_matrix(  V[:,:,0])
    B= sparse.csr_matrix(  V[:,:,1])
    dv=  DNR.dot(A) +B.dot(DMR.transpose())
    return dv

def gradient(F):
    n, m = np.shape(F)[0], np.shape(F)[1]
    DNV=sparse.identity(n)*-1+ sparse.eye(n,k=1)
    DMV =sparse.identity(m)*-1+ sparse.eye(m,k=-1)
    FS=sparse.csr_matrix(F)

    F1=DNV.dot(FS)
    F2=FS.dot(DMV)

    X=np.zeros((n,m,2))
    FA=F1.toarray()
    FB=F2.toarray()
    for i in range(n):
        for j in range(m):
            X[i][j][0]=FA[i][j]
            X[i][j][1] = FB[i][j]


    return X

def Ableitung(n):
    D=sparse.eye(n)  *-2
    O=sparse.eye(n,k=1)
    U = sparse.eye(n, k=-1)
    return O+U+D

def diffgl(F,G):
    n,m= np.shape(F)[0],np.shape(F)[1]
    dim =n*m
    FL=F.flatten(order=("F"))


    A = veclap(n, m).tocsc()
    ZW=A.dot(FL)
    g = G.flatten(order="F")
    b= A.dot(g)

    for i in range(n):                                    #"Xe" aus dem Omega(F) holen um auf andere Seite zu kriegen
        b[i]=ZW[i]
    for i in range((n*m)-n,dim):
        b[i]=ZW[i]
    for i in range(n+n-1,(dim)-n,n):
        b[i]=ZW[i]
    for i in range(n,(dim)-n,n):
        b[i]=ZW[i]

    lapF = cg(A,b)
    return lapF

def vektorfeld(F,G):
    n, m = np.shape(F)[0], np.shape(F)[1]
    dim = n * m
    FL = F.flatten(order=("F"))

    A = veclap(n, m).tocsc()
    ZW = A.dot(FL)
    b=div(vbauen(F,G),n,m)
    b=b.toarray().flatten(order="F")
    for i in range(n):  # "Xe" aus dem Omega(F) holen um auf andere Seite zu kriegen
        b[i] = ZW[i]
    for i in range((n * m) - n, dim):
        b[i] = ZW[i]
    for i in range(n + n - 1, (dim) - n, n):
        b[i] = ZW[i]
    for i in range(n, (dim) - n, n):
        b[i] = ZW[i]

    lapF = cg(A, b)
    return lapF


def vbauen(F,G):
    n,m=  np.shape(F)[0], np.shape(F)[1]
    FG=  gradient(F)
    GG= gradient(G)
    for i in range(n):
        for j in range(m):
            if norm(FG[i][j])>norm(GG[i][j]):
                GG[i][j]= FG[i][j]
    return GG




def omega(F,G,i,j):                                                                  #Teilmenge wo eingefügt werden soll
    p, q = np.shape(G)[0], np.shape(G)[1]
    return F[max(0,i-floor(p/2)):i+ceil(p/2),max(0,j-floor(q/2)):j+ceil(q/2)]

def combine(F,G,i,j,Art="Gradient"):
    p, q = np.shape(G)[0], np.shape(G)[1]
    OM=omega(F,G,i,j)
    if Art == "Gradient":
        FS=diffgl(OM,G)
    else:
        FS=vektorfeld(OM,G)


    E=FS[0]

    Z= E.reshape((np.shape(OM)[0],np.shape(OM)[1]),order="F")

    F[max(0,i+1-floor(p/2)):i-1+ceil(p/2),max(0,j+1-floor(q/2)):j-1+ceil(q/2)] = Z[1:np.shape(OM)[0]-1,1:np.shape(OM)[1]-1]
    return F



def alltogether(Großes,Kleines,i,j,Art="Gradient"):
    G = io.imread(Großes)
    RG, GG, BG = G[:, :, 0], G[:, :, 1], G[:, :, 2]
    K = io.imread(Kleines)
    RK, GK, BK = K[:, :, 0], K[:, :, 1], K[:, :, 2]                      #Extract RGB Werte into Arrays


    FB= combine(BG,BK,i,j,Art)
    FG = combine(GG, GK, i, j,Art)
    FR = combine(RG, RK, i, j,Art)                                            #Combine results

    RGB = np.dstack((FR,FG,FB))
    plt.imshow(RGB)
    plt.show()

"""
if __name__ == "__main__":
    K=io.imread(filename)
    BK,GK,RK = K[:, :, 0], K[:, :, 1], K[:, :, 2]
    L=io.imread(filename)
    BL,GL,RL = L[:, :, 0], L[:, :, 1],L[:, :, 2]
    X=io.imread("bear.jpg")
    L =io.imread("bear.jpg")
    BL,GL,RL = L[:, :, 0], L[:, :, 1],L[:, :, 2]
    K=io.imread("water.jpg")
    BK,GK,RK = K[:, :, 0], K[:, :, 1], K[:, :, 2]
    combine(BK,BL,187,165)
       K = io.imread(Großes)
    BK, GK, RK = K[:, :, 0], K[:, :, 1], K[:, :, 2]
    L = io.imread(Kleines)
    BL, GL, RL = L[:, :, 0], L[:, :, 1], L[:, :, 2]
    def channelSplit(image):
    return np.dsplit(image,image.shape[-1])
    A = veclap(n, m).tocsc()
    for i in range(n):                                    #"Xe" aus dem Omega(F) holen um auf andere Seite zu kriegen
        z=z+A.getcol(i)*FL[i]
    for i in range((n*m)-n,n*m):
        z=z+A.getcol(i)*  FL[i]
    for i in range(n,(n*m)-n,(n-1)):
        z =z+A.getcol(i) * FL[i]

    
"""