#Programa 3: DG Adam
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Marzo 2025
#=================================
#  Modulos Necesarios
#=================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from P1MinimosCuadrados import minimos_cuadrados
#=================================
#Funcion Descenso de Gradiente (ADAM)
#===================================
def DG_ADAM(epocs,dim,X,Y,alpha,Ybar):
    error = np.zeros(epocs,dtype=np.float32)
    mn = np.zeros(dim,dtype=np.float32)
    vn = np.zeros(dim,dtype=np.float32)
    g = np.zeros(dim,dtype=np.float32)
    g2 = np.zeros(dim,dtype=np.float32)
    w = np.zeros(dim,dtype=np.float32)
    beta1 = 0.80
    beta2 = 0.999
    b1 = beta1
    b2 = beta2 
    eps = 1.0e-8
    N = len(X)
    sumx = np.sum(X)
    sumy = np.sum(Y)
    sumxy = np.sum(X*Y)
    sumx2 = np.sum(X*X)
    mn[0] = -2.0*(sumy-w[0]*N-w[1]*sumx)
    mn[1] = -2.0*(sumxy-w[0]*sumx-w[1]*sumx2)
    vn = mn*mn
    for i in range(epocs):
        g[0] = -2.0*(sumy-w[0]*N-w[1]*sumx)
        g[1] = -2.0*(sumxy-w[0]*sumx-w[1]*sumx2)
        g2 = g*g
        for j in range(dim):
            mn[j] = beta1*mn[j] + (1.0-beta1)*g[j]
            vn[j] = beta2*vn[j] + (1.0-beta2)*g2[j]
        b1 *= beta1
        b2 *= beta2 
        mnn = mn/(1.0-b1)
        vnn = vn/(1.0-b2)
        fact = eps + vnn**0.5
        w -= (alpha/fact)*mnn
        Ybar2 = w[0]+w[1]*X
        error[i] = np.sum((Ybar2-Ybar)**2)
    return w,error
            
#===============================
#Programa Principal
#===============================
if __name__ =="__main__":
#=================================
#Leer datos
#=================================
    data = pd.read_csv('data.csv')
    X= np.array(data.iloc[:,0])
    Y= np.array(data.iloc[:,1])
#=================================
# Parametros
#=================================
w = np.zeros(2,dtype=np.float32)
N = len(X)
sumx = np.sum(X)
sumy = np.sum(Y)
sumxy = np.sum(X*Y)
sumx2 = np.sum(X*X)
w[1] = (N*sumxy - sumx*sumy)/(N*sumx2 - sumx*sumx)
w[0] = (sumy - w[1]*sumx)/N
Ybar = w[0] + w[1]*X

#==================================
# Descenso de gradiente (ADAM)
#===================================
w = 0.0
alpha = 2.0
epocs = 100
w,error = DG_ADAM(epocs, 2, X, Y, alpha, Ybar)
print("Error = ",[error-1])
Ybar = w[0] + w[1]*X

#==================================
# Grafica
#==================================
figure,axis = plt.subplot(2)
axis[0].scatter(X,Y)
axis[0].plot([min(X),max(X)], [min(Ybar),max(Ybar)], color='red')
axis[0].plot([min(X),max(X)], [min(Ybar),max(Ybar)], color='green')
axis[0].set_xlabel("x")
axis[0].set_ylabel("y")
axis[1].plot(error)
axis[1].set_ylabel("error")
axis[1].set_xlabel("epocs")
plt.show()

 