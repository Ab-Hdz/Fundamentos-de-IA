#Programa 11: Manejo de datos en pytorch
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Abril 2025
#==========================

#=================================
#  Modulos Necesarios
#=================================
import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#=================================
#  Bigdata debe dividirse en pequeños grupos de datos
#=================================

#=================================
#  Ciclo de entenamiento
#  for epochin range(num_epoch): 
#    # ciclo sobre todos los grupos de datos
#    for i in range(total_batches)
#=================================

#=================================
#  epoch = una evaluacion y repropagacion para todos los datos de entrenamiento
#  total_batches = número de subconjuntos de datos
#  batch_size = numero de datos de entrenamiento en cada conjunto
#  number of iterations = número de iteraciones sobre todos los datos de entrenamiento
#=================================
#  e.g : 100 samples, batch_size=20 -> 100/20=5 iteraciones for 1 epoch
#=================================

#=================================
#  DataLoader puede dividir los datos en grupos
#=================================

#=================================
#  Implementacion de base de datos tipica
#  implement_init_, _getitem_, and _len_
#=================================

#=================================
#  Hijo de Dataset
#=================================
class WineDataset(Dataset):
    
    def _init_(self):
        #=================================
        #  Inicializar, bajar datos, etc
        #  lectura con numpy o pandas
        #=================================
        #  típicos datos separados por coma
        #  delimiter = simbolo delimitador
        #  skiprows = lineas de encabezado
        #=================================
        xy = np.loadtxt('./data/wine/.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        #=================================
        #  primera columna es etiqueta de clase y el resto son caracteristicas
        #=================================
        self.x_data = torch.from_numpy(xy[:, 1:])  #grupos del 1 en adelante
        self.y_data = torch.from_numpy(xy[:, [0]])  #grupo 0
    #=================================
    #  permitir indexacion para obtener el dato 1 de dataset[i]
    #  metodo getter
    #=================================
    def _len_(self):
        return self.n_samples
    

#=================================
#  Instaciar base de datos
#=================================
dataset = WineDataset()

#=================================
#  leer caracteristicas del primer dato
#=================================
first_data = dataset[0]
features, labels = first_data
print(features, labels)

#=================================
#  Cargar toda la base con DataLoader
#  reborujar los datos (shuffle): bueno para el entrenamiento
#  num_workers: carga rapida utilizando multiples procesos
#  SI COMETE UN ERROR EN LA CARGA, PONER num_workers = 0
#=================================
train_loader = DataLoader(dataset=dataset,  # base de datos
                          batch_size=4,     # cuatro grupos
                          shuffle=True,     # reborujados
                          num_workers=2)    # dos procesos

#=================================
#  Covertir en iterador y observar un dato al azar
#=================================
dataiter= iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

#=================================
#  Ciclo de aprendizaje vacio
#=================================
num_epoch = 2 
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(train_loader):
        #=================================
        #  178 lineas, batch_size = 4, n_iters  178/4=44.5 -> 45 iteraciones
        #  corre tu proceso de aprendizaje
        #=================================
        #  Diagnostico
        if (i+1) % 5 == 0:
            print('Epoch: {epoch+1}/{num_epoch}, Step {i+1}/{n_iterations}| Inputs {inputs.shape}| Labels {labels.shape} ')
            
#=================================
#  algunas bases de datos existen en torchvision.datasets
#  e.g. MNIST, Fashion-MNIST, CIFAR10, COCO
#=================================
train_dataset =torchvision.dataset.MNIST(root='./data',
                                         train=True,
                                         transform=torchvision.transform.ToTensor(),
                                         download=True)

train_loader = DataLoader(dataset=train_dataset,
                                          batch_size=3,
                                          shuffle=True)

#=================================
#  look at one random sample
#=================================
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)








































































