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


def veclap(n,m):             #Returns Vektorisierten Laplace Operator für Matrizen der Größe n,m
    Dn=Ableitung(n)
    Dm=Ableitung(m)
    In=  sparse.identity(n)
    Im = sparse.identity(m)
    lap= (sparse.kron(Im,Dn)+sparse.kron(Dm,In))
    return lap

def div(V,n,m):                                              #return Divergenz von Vektorfeld v
    DNR=  sparse.eye(n, k=-1)*-1+ sparse.identity(n)
    DMR=  sparse.eye(m, k=-1)*-1+ sparse.identity(m)
    A=sparse.csr_matrix(  V[:,:,0])
    B= sparse.csr_matrix(  V[:,:,1])
    dv=  DNR.dot(A) +B.dot(DMR.transpose())
    return dv

def gradient(F):                                               #return vorwärts gradient von F
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

def Ableitung(n):               #return "Dn" Sparse Matrix size N, Zweite Ableitung Operator
    D=sparse.eye(n)  *-2
    O=sparse.eye(n,k=1)
    U = sparse.eye(n, k=-1)
    return O+U+D

def diffgl(F,G):                                         #solve lap(F)=lap(G) with nebenbedingunen (Gleich auf dem Rand)
    n,m= np.shape(F)[0],np.shape(F)[1]
    dim =n*m
    FL=F.flatten(order=("F"))


    A = veclap(n, m).tocsc()
    ZW=A.dot(FL)
    g = G.flatten(order="F")
    b= A.dot(g)

    for i in range(n):                                    #"Bekannten" auf die rechte seite bringen
        b[i]=ZW[i]
    for i in range((n*m)-n,dim):
        b[i]=ZW[i]
    for i in range(n+n-1,(dim)-n,n):
        b[i]=ZW[i]
    for i in range(n,(dim)-n,n):
        b[i]=ZW[i]

    lapF = cg(A,b,maxiter=100000)
    return lapF

def vektorfeld(F,G):                                 #solve lap(F)= div v wie in aufgabe beschrieben
    n, m = np.shape(F)[0], np.shape(F)[1]
    dim = n * m
    FL = F.flatten(order=("F"))

    A = veclap(n, m).tocsc()
    ZW = A.dot(FL)
    b=div(vbauen(F,G),n,m)
    b=b.toarray().flatten(order="F")
    for i in range(n):                              # "Bekannten" auf die rechte seite bringen
        b[i] = ZW[i]
    for i in range((n * m) - n, dim):
        b[i] = ZW[i]
    for i in range(n + n - 1, (dim) - n, n):
        b[i] = ZW[i]
    for i in range(n, (dim) - n, n):
        b[i] = ZW[i]

    lapF = cg(A, b, maxiter=100000)
    return lapF


def vbauen(F,G):                                       #build vektorfeld nach aufgabenstellung
    n,m=  np.shape(F)[0], np.shape(F)[1]
    FG=  gradient(F)
    GG= gradient(G)
    for i in range(n):
        for j in range(m):
            if norm(FG[i][j])>norm(GG[i][j]):
                GG[i][j]= FG[i][j]
    return GG




def omega(F,G,i,j):                                                  #Ausschnitt des Bildes wo eingefügt werden soll ausgeben
    p, q = np.shape(G)[0], np.shape(G)[1]
    return F[i:i+p,j:j+q]

def combine(F,G,i,j,Art="Gradient"):                            #Insert G into F at Location i,j (nxmx1 dimensionale Matrizen)
    p, q = np.shape(G)[0], np.shape(G)[1]
    OM=omega(F,G,i,j)

    if Art=="Nichts":
        F[i+1:i+p-1,j+1:j+q-1] = G[1:np.shape(OM)[0]-1,1:np.shape(OM)[1]-1]
        return F


        
    if Art == "Gradient":
        FS=diffgl(OM,G)
    else:
        FS=vektorfeld(OM,G)


    E=FS[0]                                                                        #

    Z= E.reshape((np.shape(OM)[0],np.shape(OM)[1]),order="F")                      #zurück in matrix form

    F[i+1:i+p-1,j+1:j+q-1] = Z[1:np.shape(OM)[0]-1,1:np.shape(OM)[1]-1]          #Ergebnisse einsetzen
    return np.clip(F,0,255)                                                                    #Matrix hat auch Werte > 255, daher runterclippen auf 0-255



def alltogether(Großes,Kleines,i,j,Art="Gradient",Title=""):                  #Bilder in R,G,B Matrizen aufteilen, einzelne Komponenten kombinieren und wieder zusammen stecken
    G = io.imread(Großes)
    G=G.astype("int64")
    RG, GG, BG = G[:, :, 0], G[:, :, 1], G[:, :, 2]
    K = io.imread(Kleines)
    K = K.astype("int64")
    RK, GK, BK = K[:, :, 0], K[:, :, 1], K[:, :, 2]                      #Extract RGB Werte into Arrays


    FB= combine(BG,BK,i,j,Art)
    FG = combine(GG, GK, i, j,Art)
    FR = combine(RG, RK, i, j,Art)                                            #Combine results

    RGB = np.dstack((FR,FG,FB))
    plt.imshow(RGB)                                                          #show results
    plt.title(Title)
    plt.show()


if __name__ == "__main__":
    alltogether("water.jpg","bear.jpg",100,30,Title="Laplace")
    alltogether("water.jpg", "bear.jpg", 100,30, Art="B",Title="Divergenz")
    alltogether("water.jpg","bear.jpg",100,30,"Nichts","Keine Bearbeitung")
    alltogether("bird.jpg","plane.jpg",50,400,Title="Laplace")
    alltogether("bird.jpg", "plane.jpg",50, 400, Art="B", Title="Divergenz")
    alltogether("bird.jpg","plane.jpg",50,400,"Nichts","Keine Bearbeitung")
