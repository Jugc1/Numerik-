import matplotlib.pyplot as plt
from math import floor
from math import ceil
import math
import matplotlib
import numpy as np
from numpy.linalg import norm
import skimage
from skimage import io
import os
from scipy import sparse
from scipy.sparse.linalg import cg



def mittelwert(Index, Bilder):
    Sum = np.zeros([28, 28], dtype=float)
    N = len(Index)
    for i in Index:
        Sum += Bilder[i, :, :]

    return Sum / N


def varianz(Index, Bilder, b):
    Sum = np.zeros([28, 28], dtype=float)
    N = len(Index)
    for i in Index:
        Z = Bilder[i, :, :] - b
        Z = Z.astype(float)

        Z = np.multiply(Z, Z)
        Sum += Z
    return (1 / N) * Sum


def cov(Index, Bilder):
    b = mittelwert(Index, Bilder)
    Y = np.zeros([784, len(Index)], dtype=float)
    for j, i in enumerate(Index):
        s = Bilder[i, :, :] - b
        s = s.astype(float)
        Y[:, j] = s.flatten(order="F")

    return np.nan_to_num(Y)


def UR(Index, Bilder, d):
    b = mittelwert(Index, Bilder)
    Y = cov(Index, Bilder)
    u, s, vh = np.linalg.svd(Y)
 
    A = u[:, :d]
    return A, b


def proj(A, b, Bild):
    Ag = A.transpose()
    s = Bild - b
    t = Ag.dot(s.flatten(order="F"))
    return t


def daten(labs, max, s):
    List = []
    z = 0
    for j, i in enumerate(labs):

        if i == s:
            List.append(j)
            z += 1
        if z == max:
            break
    return List


def kmeans(daten,Mittelpunkte):       #Mittelpunkte liste von vektoren
    n,m=daten.shape
    p,q=Mittelpunkte.shape

    
    for s in range(100):

        Klassen = [[] for _ in range(q)]
        for i in range(n):

            p=[]
            x= daten[i,:]
            for j in range(q):
                y=norm(Mittelpunkte[j,:]-x)**2
                p.append(y)

            Klassen[np.argmin(p)].append(x)




        for i in range(q):
            ck=len(Klassen[i])

            if ck!=0:
                Sum= np.zeros((2,))
                for l in Klassen[i]:
                    Sum+=l

                Mittelpunkte[i,:]=(Sum/ck)



    return Klassen,Mittelpunkte           



                
            

        


def Mitte(Index,S1):
    N=len(Index)
    return Summe(Index,S1)/N



def Summe(Index, S1):
    Sum = np.zeros([2, ], dtype=float)
    for i in Index:
        Sum += S1[i, :]
    return Sum


def makeset(imgs, L):

   
    A, c = UR(L, imgs, 2)

    New = np.zeros([len(L),2], dtype=float)

    for j, i in enumerate(L):
        t = proj(A, c, imgs[i, :, :])

        
        New[j,:] = t
    return New

def show2(labs,imgs):
    for s in range(10):
        List = daten(labs, 100, s)
        b = mittelwert(List, imgs)
        plt.imshow(b, cmap="gray")
        plt.title(f"{s} Mittelwert")
        plt.show()
        plt.imshow(varianz(List, imgs, b), cmap="gray")
        plt.title(f"{s} Varianz")
        plt.show()


def show3(labs,imgs):
    I = [i for i in range(1000)]
    u, s, vh = np.linalg.svd(cov(I, imgs))
    EW = s[:50]
    EW = np.diag(EW)
    plt.imshow(EW, cmap="gray")
    plt.title("Eigenwerte")
    plt.show()
    

    A, c = UR(I, imgs, 5)

    plt.imshow(c, cmap="gray")
    plt.title("Mittelwert Aufgabenteil 3")
    plt.show()

    for i in range(1, 6):
        plt.imshow(A[:, i - 1:i].reshape([28, 28], order="F"), cmap="gray")
        plt.title(f"Hauptkomponente {i}")
        plt.show()

    for i in range(4):
        t = proj(A, c, imgs[i, :, :])
        x = A.dot(t).reshape([28, 28], order="F") + c
        plt.imshow(imgs[i, :, :], cmap="gray")
        plt.title(f"Testbild {i}")
        plt.show()
        plt.imshow(x, cmap="gray")
        plt.title(f"Projektion {i}")
        plt.show()
def show4(labs,imgs):
    L1 = daten(labs, 1000, 0)
    L2 = daten(labs, 1000, 1)
    L = L1 + L2
    A, c = UR(L, imgs, 2)
    S = makeset(imgs, L)

    M1 = Mitte([i for i in range(100)], S)
    M2 = Mitte([i for i in range(1000, 1100)], S)

    M = np.column_stack((M1, M2))

    Klassen, Mittelpunkte = kmeans(S, M)

    K0 = np.array(Klassen[0])
    K1 = np.array(Klassen[1])

    plt.scatter(K0[:, 0], K0[:, 1], color="red")
    plt.scatter(K1[:, 0], K1[:, 1], color="green")

    plt.scatter(Mittelpunkte[0, 0], Mittelpunkte[0, 1], color="yellow")
    plt.scatter(Mittelpunkte[1, 0], Mittelpunkte[1, 1], color="blue")
    plt.title("K-Means Clustering Vergleich 0(rot)/1(grün)")
    plt.show()

    EI = np.zeros([100, 2], dtype=float)
    NE = np.zeros([100, 2], dtype=float)
    for i in range(100):
        t1 = proj(A, c, imgs[L1[i], :, :])
        t2 = proj(A, c, imgs[L2[i], :, :])
        EI[i, :] = t1
        NE[i, :] = t2

    M1 = Mittelpunkte[0, :]
    M2 = Mittelpunkte[1, :]
    richtig0 = 0
    falsch0 = 0
    for i in range(100):
        if norm(M1 - EI[i]) <= norm(M2 - EI[i]):
            richtig0 += 1
        else:
            falsch0 += 1

    richtig1 = 0
    falsch1 = 0
    for i in range(100):
        if norm(M2 - NE[i]) ** 2 <= norm(M1 - NE[i]) ** 2:
            richtig1 += 1
        else:
            falsch1 += 1
    fig, ax = plt.subplots()
    table_data = [
        ["Richtige Klassifizierungen der 0", richtig0],
        ["Falsche Klassifizierungen der 0", falsch0],
        ["Richtige Klassifizierungen der 1", richtig1],
        ["Falsche Klassifizierungen der 1", falsch1]

    ]
    table = ax.table(cellText=table_data, loc='center')
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')
    plt.title("Testklassifizierungen 0 und 1")
    plt.show()

if __name__ == "__main__":
    
    imgs = np.fromfile("train-images-idx3-ubyte", dtype=np.uint8)
    imgs = np.reshape(imgs[16:], [-1, 28, 28])
    labs = np.fromfile("train-labels-idx1-ubyte", dtype=np.uint8)
    labs = labs[8:]
    """
    show2(labs,imgs)
        
    show3(labs, imgs)
    """
    show4(labs, imgs)