#Programa 7: Diferenciacion automatica Autograd
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Abril 2025
#==========================

import torch
#=========================
# requires_grad = True -> genera funciones gradiente
# para las operaciones que se hacen con ese tensor
#=========================
x = torch.rand(3, requires_grad=True)
y = x + 2
#==========================
# y = y(x) tiene un grad_fn asociado
#==========================
print(x)
print(y)
print(y.grad_fn)

#=========================
# z = z(y) = z(y(x))
#=========================
z = y * y * 3 #Multiplica las entradas del tensor
print(z)
z = z.mean() #Calcula el promedio delas entradas
print(z)

#=======================
# Calculo del gradiente con retropropagcion
#========================
z.backward()
print(x.grad) #dz/dx

#======================
# Torch.aotogard se basa en regla de la cadena
#======================
x = torch.rand(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2
print(y)
print(y.shape)

#===================
# Evaluar "gradiente" dy/dx en v
#===================
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

#=======================
# Decirle a un tensor que deje generar gradientes
#=======================
# - x.requires_grad_(False)
# - x.detach()
# - envolverlo con 'with torch.no_grad():
# .requires_grad_( . . .) cambia la bandera
#======================
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)

#======================
# con gradiente
#=====================
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
x = torch.randn(3, requires_grad=True)
print(a.requires_grad)
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)

#====================
# sin gradientes
#===================
b = a.detach()
print(b.requires_grad)

#=====================
# con envoltura que le quita el gradiente 
#=====================
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

#=====================
# backward() acumula el gradiente en .grad
# .zero_() limpia el gradiente antes de comenzar
#=====================
weights = torch.ones(4, requires_grad=True)
print(weights)

#===================
# Epoch: paso de optimización
#==================
for epoch in rage(3):
  # ejemplito
  model_output = (weights*3).sum()
  model_output.backward()
  print(weights.grad)
  # optimizacion: encontrar nuevos coeficientes
  with torch.no_grad():
      weights -= 0.1 * weights.grad
  # reinicializa el gradiente a cero (importante)
  weigdts.grad.zero_()
model_output = (weights*3).sum()
print(weights)
print(model_output)

#=====================
# Optimizer tiene metodo zero_grad()
# optimizer = torch.optim.SGD([weights], lr=0.1)
# SGD = Stochastic Gradient Descent
# During trainind:
# optimizer.step()
# optimizer.zero_grad()
      