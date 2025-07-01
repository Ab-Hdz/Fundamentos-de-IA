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
        super(MultiHeadAttention, self).__init__():
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

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
class PositionWiseFeedForward(nn.Module):
    #=================================
    #  Constructor con elementos necesarios
    #=================================
    def __init__(self, d_model, d_ff):
        super(PositiocWiseFeedForward, self).__int___()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    #=================================
    #  Red feed forwars
    #=================================
   def forward(self, x):
       return self.fc2(self.relu(self.fcl(x)))
   
#=================================
#  Codificación posicional
#=================================
class PositionalEncoding(nn.Module):
    #=================================
    #  Connstructor con elementos necesarios
    #=================================
    def __init__(self, d_model, max_seq_legth):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*-(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position *div_term)
        pe[:, 1::2] = torch.cos(position *div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    #=================================
    #  Añadir la posición
    #=================================
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
#=================================
#  Codificador
#=================================
class EncodeLayer(nn.Module):
    #=================================
    #  Constructor con los elementos necesarios
    #=================================
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttetion(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    #=================================
    #  Algoritmo codificador
    #=================================
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
#=================================
#  Decodificador
#=================================
class DecoderLayer(nn.Module):
    #=================================
    #  Constructor de modulos necesarios
    #=================================
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecodreLayer, self).__init__()
        self.self_attn = MultiHeadAttetion(d_model, num_heads)
        self.cross_attn = MultiHeadAttetion(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    #=================================
    #  Algoritmo decodificador
    #=================================
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
#=================================
#  Red Transformer
#=================================
class Transformer(nn.Module):
    #=================================
    #  Construtor y elementos necesarios
    #=================================
    def __init__(self, d_model, num_heads, d_ff, dropout,src_vocab_size, tgt_vocab_size, max_seq_length, num_layers):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout)for _ in range(num_layers)])
        self.fc = nn.Linear(d_model,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    #=================================
    #  Generar mascarillas (para bloquear atención)
    #=================================
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1,seq_length,seq_length)diagonal=1)).bool()
        tgt_mask = tgt_mask & ~nopeak_mask
        return src_mask, tgt_mask
    
    #=================================
    #  Algoritmo Tranformer
    #=================================
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_ecoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_ecoding(self.decoder_embedding(tgt)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

#=================================
#  Programa Principal
#=================================
if __name__ == "__main__":
    src_vocab_size = 5000    #Num de palabras de entrada
    tgt_vocab_size = 5000    #Num de palabras para comprar
    d_model = 512            #Dimensión de los embeddings
    num_heads = 8            #Canales de atención
    num_layers = 6           #Capas
    d_ff = 2048              #Dimensión de la red feed-forward
    max_seq_length = 100     #Maxima longitud de las frases
    dropout = 0.1            #Fracción de reducción de dimensiones
    
#=================================
#  Crear la red transformer
#=================================
transformer = Transformer(src_vovab_size,tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout)

#=================================
#  Generar tabla con datos al azar
#=================================
src_data = torch.randint(1, src_vocab_size,(64, max_seq_length))    #(batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size,(64, max_seq_length))    #(batch_size, seq_length)
  
#=================================
#  Error medido con la entropia
#================================= 
criterion = nn.CrossEntropyLoss(ignore_index=0)

#=================================
#  Metodo de Adam para el descenso de gradiente
#=================================
optimizer = optim.Adam(transformer.parameters(),lr=0.0001,betas=(0.9, 0.98),eps=1e-9)

#=================================
#  Correr entrenamiento
#=================================
transformer.train()
for epoch in range(10):
    #=================================
    #  Limpiar el gradiente
    #=================================
    optimize.zero_grad()
    #=================================
    #  Aplicar transformer a los datos y obtener resultados
    #=================================
    output = transformer(src_data, tgt_data[:, :-1])
    #=================================
    #  Cálculo del error entre frases de salida y frases reales
    #=================================
    loss = criterion(output.contiguous().view(-1,tgt_vocab_size),tgt_data[:,1:].contiguous().view(-1))
    #=================================
    #  Ca´lculo del gradiente
    #=================================
    loss.backward()
    #=================================
    #Descenso de gradiente
    #=================================
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')