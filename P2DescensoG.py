
# Programa 2:  Descenso de Gradiente para Minimos Cuadrados
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Febrero 2025
#=================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from P1MinimosCuadrados import minimos_cuadrados
#===================================
# Descenso de gradiente simple
#=================================== 
def desensoG(epocs, X,Y,alpha):
    w0 = 0.0 
    w1 = 0.0 
    N = len(X)
    sumx = np.sum(X)
    sumy = np.sum(Y)
    sumxy = np.sum(X*Y)
    sumx2 = np.sum(X*X)
    for i in range(epocs):
        Gradw0 = -2.0*(sumy-w0*N-w1*sumx)
        Gradw1 = -2.0*(sumxy-w0*N-w1*sumx2)
        w0 -= alpha*Gradw0
        w1 -= alpha*Gradw1
        return w0,w1
#=================================
#Funcion Principal
#=================================
if __name__ =="__main__":
#=================================
        
#==========================================
# Programa Principal
#========================================       
#Leer datos
#=================================
    data = pd.read_csv('data.csv')
    X= np.array(data.iloc[:,0])
    Y= np.array(data.iloc[:,1])

#==================================
#Minimos Cuadrados
#==================================
    Ybar,w0,w1 = (minimos_cuadrados(X, Y))
#===================================
# Descenso de gradiente
#===================================
w0 = 0.0
w1 = 0.0
alpha = 0.025
epocs = 100
w0,w1 = desensoG(epocs, X,Y,alpha)
Ybar2 = w0 + w1*X
#===================================
# Grafica
#===================================
plt.scatter(X, Y)
plt.rcParams['figure.figsize']= (12.0, 9.0)
plt.plot([min(X), max(X)], [ min(Ybar), max(Ybar)], color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

    