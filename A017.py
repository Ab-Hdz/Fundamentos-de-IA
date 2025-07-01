#Programa 17: Transformer con pytorch
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# mayo 2025
#==========================

#=================================
#  Modulos Necesarios
#=================================
import torch 
import torch.nn as nn
import torch.optim import optim
import torch.utils.data as data
import math
import copy

#=================================
#  Celula de atención (multiples)
#=================================
class MultiHeadAttention(nn.Module):
    #=================================
    #  Constructor
    #=================================
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__:
            assert d_model % num_heads == 0, "d_model must be ddivisible by num_heads"
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

#=================================
#  Producto escalar escalado
#=================================
def scaled_dot_product_attetion(self, Q, K V, mask=None):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output

#=================================
#  Crear subconjuntos
#=================================
def split_heads(self, x):
    batch_size, seq_length, d_k = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

#=================================
#  Combinar y transponer subconjuntos
#=================================
def combine_heads(self, x):
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1, 2).configuous().view(batch_size, seq_length, self.d_model)

#=================================
#  Red de la célula de atención
#=================================
def forward(self, Q, K , V, mas=None):
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))
    attn_output = self.scaled_dot_product_attetion(Q, K, V, mask)
    output = self.W_o(self.combine_heads(attn_output))
    return output

#=================================
#  Red neuronal clásica (fedd-forward)
#=================================
class























    











