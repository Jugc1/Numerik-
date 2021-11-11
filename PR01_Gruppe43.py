import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import io
import os 


def n_mittel(Matrix, Filter=None, var=None):
    s = Matrix.shape
    n = s[0]
    m = s[1]
    Sum = 0
    if Filter == "G":
        W = gauß_filter(var, n, m)
    else:
        W = rechteck_filter(n, m)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Sum += W(i, j) * Matrix[i - 1, j - 1]

    return Sum


def n_median(Matrix, Filter=None, var=None):
    s = Matrix.shape
    n = s[0]
    m = s[1]
    Sort = np.sort(Matrix, axis=None)
    SortI = np.argsort(Matrix, None)
    L = []
    for i, j in enumerate(SortI):
        a = round(j / m)
        b = j % m
        L.append([a, b])

    if Filter == "G":
        W = gauß_filter(var, n, m)
    else:
        W = rechteck_filter(n, m)

    A = summing(n, m, W, Sort, L)

    Sum = A[0]
    index = A[1]

    if Sum >= 0.4999:
        return Sort[index]
    else:
        a = Sort[index]
        b = Sort[index + 1]
        return (a + b) / 2

def bila(UMatrix,Matrix, vars,varr,k,l,s):
    a= vars**2
    b = varr**2
    wr = lambda x: np.exp((-(x**2)/(2*b)))
    ws = lambda x, y: np.exp(-(x ** 2 + y ** 2))/ (2 * a)
    g = UMatrix.shape
    n = g[0]
    m = g[1]
    SumL=0
    for i in range(1, n+1):
        for j in range(1, m+1):
            
            SumL+=(ws(k-(k-1-s+i),l-(l-1-s+j))*wr((np.int64(Matrix[k-1][l-1]))-(np.int64(UMatrix[i-1][j-1]))))

    
    Sum= 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            Sum+=(((ws(k-(k-1-s+i),l-(l-1-s+j))*wr(np.int64(Matrix[k-1][l-1])-np.int64(UMatrix[i-1][j-1])))) /SumL ) * np.int64(UMatrix[i-1][j-1])  #np.int64(Matrix[k-1-s+i][l-1-s+j])
    return Sum
def summing(n, m, Weight, Sort, Sort2):
    Sum = 0
    for i, j in enumerate(Sort):
        Sum += Weight(Sort2[i][0], Sort2[i][1])
        if Sum >= 0.49999:
            return ([Sum, i])


def gauß_filter(var, n, m):
    M = 3 * round(var)
    W1 = lambda k, l: 1/(2*math.pi*var**2)*( np.exp((-((k-(n-1)/2) ** 2 + (l-(m-1)/2) ** 2)) / (2 * var ** 2) ))if (abs((k-(n-1)/2)) <= M and abs((l-(m-1)/2)) <= M) else 0
    S = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S += W1(i, j)
    W = lambda k, l:1/(2*math.pi*var**2)* (np.exp((-((k-(n-1)/2) ** 2 + (l-(m-1)/2) ** 2)) / (2 * var ** 2)) / S) if (abs((k-(n-1)/2)) <= M and abs((l-(m-1)/2)) <= M) else 0
    return W


def rechteck_filter(n, m):
    W = lambda k, l: 1 / (n * m) if k <= n and l <= m else 0
    return W


def test(Anzahl):
    maxme = 0
    maxmi = 0
    for i in range(Anzahl):
        arr = np.random.randint(100, size=(np.random.randint(1, 100), np.random.randint(1, 100)))
        a = n_median(arr)
        b = n_mittel(arr)
        maxme = max(maxme, a - np.median(arr))
        maxmi = max(maxmi, a - np.mean(arr))
    return ("Abweichung Median ", maxme, "Abweichung Mittel ", maxmi)





def input(filename):
    X=io.imread(filename)
    return X

def pad(Array, s, mode="edge"):
    X=np.pad(Array, s, mode)
    return X

def g_mittel(Matrix, Filter=None, Var=None, erweiterung="edge", s=2):
    P=pad(Matrix,s, erweiterung)
    G = Matrix.shape
    n = G[0]
    m = G[1]
    for i in range(0, n ):
        for j in range(0, m ):
            U = P[max(0,i-s):i+s,max(0,j-s):j+s]
            Matrix[i][j]= n_mittel(U,Filter,Var)
    return Matrix

def g_median(Matrix, Filter=None, Var=None, erweiterung="edge", s=2):
    P=pad(Matrix,s, erweiterung)
    G = Matrix.shape
    n = G[0]
    m = G[1]
    for i in range(0, n ):
        for j in range(0, m ):
            U = P[max(0,i-s):i+s,max(0,j-s):j+s]
            Matrix[i][j]= n_median(U,Filter,Var)
    return Matrix
def BilateralFilter(Matrix,varr,vars,erweiterung="edge",s=2):
    P = pad(Matrix, s, erweiterung)
    G = Matrix.shape
    n = G[0]
    m = G[1]
    for i in range(0, n ):
        for j in range(0, m ):
            U = P[max(0,i-s):i+s,max(0,j-s):j+s]
            
            Matrix[i][j] = bila(U,Matrix,vars,varr,i+1,j+1,s)

    return Matrix
if __name__ == "__main__":
    print(test(100))
    Gauß= gauß_filter(10,100,100)
    ExampleFilter= np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            ExampleFilter[i][j]=Gauß(i,j)
    
    plt.imshow(ExampleFilter, cmap="gray", interpolation="none")
    plt.title("Gaussgewichte")
    plt.show()
    B1 = input("B1.png")
    BMI = g_median(B1,Filter="G",Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B1 Median Gauss")
    plt.show()
    
    B2 = input("B2.png")
    BMI = g_median(B2,Filter="G",Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B2 Median Gauss")
    plt.show()
    
    C = input("C.png")
    BMI = g_median(C,Filter="G",Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("C Median Gauss")
    plt.show()
    
    BMI = g_median(B1,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B1 Median")
    plt.show()

    BMI = g_median(B2,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B2 Median")
    plt.show()
    
    BMI = g_median(C,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("C Median")
    plt.show()
    
    BMI = g_mittel(B1,Filter="G",Var=3, erweiterung="symmetric",s=1)
    plt.imshow(BMI, cmap="gray")
    plt.title("B1 Mittel Gauss")
    plt.show()
    
    BMI = g_mittel(B2,Filter="G",Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B2 Mittel Gauss")
    plt.show()
    
    BMI = g_mittel(C,Filter="G",Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("C Mittel Gauss")
    plt.show()
    
    BMI = g_mittel(B1,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B1 Mittel")
    plt.show()
    
    BMI = g_mittel(B2,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B2 Mittel")
    plt.show()
    
    BMI = g_mittel(C,Filter=None,Var=3, erweiterung="symmetric",s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("C Mittel")
    plt.show()
    
    BMI = BilateralFilter(B1,erweiterung="edge",varr=75,vars=3,s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B1 Bilateral")
    plt.show()
    
    BMI = BilateralFilter(B2,erweiterung="edge",varr=75,vars=3,s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("B2 Bilateral")
    plt.show()
    
    BMI = BilateralFilter(C,erweiterung="edge",varr=75,vars=3,s=2)
    plt.imshow(BMI, cmap="gray")
    plt.title("C Bilateral")
    plt.show()
    
