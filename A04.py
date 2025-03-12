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
# Funcion de activacion ReLU
#==================================
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
#=================================
# Evaluar red
#=================================
def forward_prop(W1,b1,W2,b2,X):
    