#Programa 20: Agrupamientos usando conjuntos de particulas
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Junio 2025
#==========================

#=================================
#  Modulos Necesarios
#=================================
import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm

plot = True

#=================================
#  Leer datos (hoja de datos)
#=================================
data_points = pd.read_csv('iris.txt', sep=',', header=None)

#=================================
#  Pasar columna 4 (comienza en 0) a un arreglo de numpy (dataframe a numpy)
#=================================
clusters = data_points[4].values

#=================================
#  Remover columna 4 de los datos (metodo drp de pandas)
#=================================
data_points = data_points.drop([4], axis=1)

#=================================
#  Usar columnas 0 y 1 como (x,y) para graficar puntos en 20
#=================================
if plot:
    data_points = data_points[[0, 1]]

#=================================
#  Convierte a arreglo de numpy 2d
#=================================
data_points = data_points.values

#=================================
#  Algoritmo PSO-Clusterin
#=================================
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)

#=================================
#  Mapeo de colores a elementos de los grupos
#=================================
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
#print('Actual classes = ', clusters)