# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 02:37:57 2025

@author: abiga
"""
#Programa 15: Ejemplo de red neuronal convolucional
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#=======================
# Configuracion del GPU
#=======================
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#====================
# Hiper - parámetros
#====================
num_epochs=10		#Iteraciones sobre los datos de entrenamiento
batch_size=4		#Subconjuntos de datos
learning_rate=0.001	#Tasa de aprendizaje

#============================================================
#Definir pre-procesamiento de datos (transformacion)
#============================================================
# La base de datos tiene imagenes PILImage en el rango [0,1]
# Las transformamos a Tensores de rango normalizado [-1,1]
#============================================================
transform=transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5.0.5.0.5),(0.5.0.5.0.5))])

#==================================================================================
# CIFAR10: 60000 32X32  imagenes a color en 10 clases, con 6000 imagenes por clase
#==================================================================================
train_dataset=torchvision.datasets.CIFAR10(root='./data', train=True,
					   download=True, transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root='./data', train=False,
					   download=True, transform=transform)
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,
					 shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,
					 shuffle=False)

#======================
# Objetos a clasificar
#======================
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

#=========================
# Graficar con matplotlib
#=========================
def imshow(img):
	img=img/2+0.5 #unnormalize
	npimg=img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

#========================================
# Obtener algunas imagenes para entrenar
#========================================
dataiter=iter(train_loader)
images,labels=next(dataiter)

#===============================
# Mostrar contenido de imagenes
#===============================
imshow(torchvision.utils.make_grid(images))

#============================
# Red neuronal convolucional
#============================
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet,self).__init__()
		self.conv1=nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		self.fc1=nn.Linear(16*5*5,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)

	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))
		#reorganizar el tensor
		x=x.view(-1,16*5*5)
		# redes lineales + relu
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		#red lineal
		x=self.fc3(x)
		return x

#============================
# Correr el modelo en el Gpu
#============================
model=ConvNet().to(device)

#=======================================================================
# usar crossentropy como costo y gradiente estocastico como optimizador
#=======================================================================
criterion=nn.CrossEntropyLosss()
optimizer=torch.opim.SGD(model.parameters(),lr=learning_rate)

#=============================
# iteraciones (entrenamiento)
#=============================
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
	for i, (images,labels) in enumerate(train_loader):
		images=images.to(device)
		labels=labels.to(device)
		outputs=model(images)
		loss=criterion(outputs,labels)
		optimizer.zer_grad()
		loss.backward()
		optimizer.step()
		if (i+1) %20000==0:
			print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')
print('Entrenamiento completo')

#==========================================
# guardar resultado del modelo (parámetros)
#==========================================
PATH = './cnn.pth'
torch.save(model.state_dict(),PATH)

#==================
# Probar el modelo
#==================
with torch.no_grad():
	n_correct=0
	n_samples=0
	n_class_correct=[0 for i in range(10)]
	n_class_samples=[0 for i in range(10)]
	for images, labels in test_loader:
		images=images.to(device)
		labels=labels.to(device)
		outputs=model(images)
		_,predicted=torch.max(outputs,1)
		n_samples += labels.size(0)
		n_correct += (predicted==labels).sum().item()

		for i in range(batch_size):
			label=labels[i]
			pred=predicted[i]
			if(label==pred):
				n_class_correct[label]+=1
			n_class_samples[label]+=1

	acc=100.0*n_correct/n_samples
	print(f'Precision del modelo: {acc}%')

	for i in range(10):
		acc=100.0*n_class_correct[i]/n_class_samples[i]
		print(f'Precision de {classes[i]}: {acc}%')