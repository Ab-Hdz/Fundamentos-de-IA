#Programa 12: Transformers
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Mayo 2025
#==========================

'''
Transformaciones pueden ser aplicadas a imagenes PIL, tensores, ndarrays
o datos comunes durante la creación de la base de datos comunes durante la creacion de la base de datos

lista completa de transformaciones ya programadas:
https://pytorch.org/docs/stable/torchvision/transforms.html

En imagenes
=================================
CenterCrop, Gayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

En Tensores
=================================
LinearTransformation, Normalize, RandomFrasing

Conversiones
=================================
ToPILImage: convertir de tensor a numpy ndarray
ToTensor : de numpy.ndarray a PILImage

Generico
=================================
Usar Lambda

Comunes
=================================
Escribir tu propio objeto (clase)

Componer (compose) multiples transformaciones
=================================
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
'''

#=================================
#  Modulos Necesarios
#=================================
import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np

#=================================
#  Clase WineDataset hija de Dataset
#=================================
class WineDataset(Dataset):
    #=================================
    #  Constructor
    #=================================
    def _init_(self, transform=None):
        xy = np.load('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        #note que no convertimos en tensor aquí
        self.x_data =xy[:, 1:]
        self.y_data =xy[:, [0]]
        
        self.transform = transform
        
    #=================================
    #  Metodos para obtener datos
    #=================================
    def _getitem_(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    #=================================
    #  Tamaño de conjunto de datos
    #=================================
    def _len_(self):
        return self.n_samples
    
#=================================
#  Transformaciones comunes
#=================================

#=================================
#  De numpy a tensor pytorch
#=================================
class ToTensor:
    def _call_(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
#=================================
#  Escalar datos (multiplicarlos por una constante)
#=================================
class MulTransform:
    
    def _init_(self, factor):
        self.factor= factor
        
    def _call_(self, samples):
        inputs, targets = sample 
        inputs *= self.factor 
        return inputs, targets
    
#=================================
#  Programa principal
#=================================
if __name__ == "__main__":
    print('Sin transformacion')
    dataset = WineDataset()
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
    
    print('\nTransformado en tensior')
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
    
    print('\nCon Transformacion a tensor y multiplicacion')
    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
























