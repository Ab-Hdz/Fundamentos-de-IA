#Programa 4: Red neuronal de dos capas
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
 
#=================================
#Inicializacion
#=================================
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2
#=================================
#Funcion de activacion ReLU
#=================================
def ReLU(Z):
    return np.maximum(Z,0)
#=================================
# Funcion de activacion softmax
#==================================
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
#=================================
# Evaluar red
#=================================
def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) +b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

#===============================
# Derivada de la ReLU
#===============================
def ReLU_deriv(Z):
    return Z>0
#===============================
# Clasificacion de salidas
#===============================
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
#===============================
# Calculo del gradiente
#===============================
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    