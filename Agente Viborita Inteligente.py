#Programa : Agente Viborita Inteligente
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Junio 2025
#==========================

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
JUEGOS_AL_AZAR = 80

#=================
# Clase Agente
#================
class Agent:
      #=============================
      # Consteuctor:
      #  model -red neuronal
      #    trainer . optimizador
      #==============================

    def __init__(self):
        self.n_games = 0
        self.epsilon = JUEGOS_AL_AZAR  # juegos al azar
        self.gamma = 0.9 # tasa de descuento
        self.memory = deque(maxlen=MAX_MEMORY) # pila finita popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
      #=============================
      # Estado del agente
      #=============================

    def get_state(self, game):
        head = game.snake[0]
        #=====================
        # pixel 20x20
        #=====================
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
       
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #==================
            # Peligro en frente
            #==================
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #=====================
            # peligor a la derecha
            #=====================
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #=======================
            # peligor a la izquierda
            #========================
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
           
            #=============================
            # Dirección de movimiento
            #=============================
            dir_l,
            dir_r,
            dir_u,
            dir_d,
           
            #======================
            # posición de la comida
            #======================
            game.food.x < game.head.x,  # comida a la izquierda
            game.food.x > game.head.x,  # comida a la derecha
            game.food.y < game.head.y,  # comida arriba
            game.food.y > game.head.y  # comida abajo
            ]
        #==============================================
        # Regresa estado convertido a enteros ( 0 o 1)
        #==============================================
        return np.array(state, dtype=int)
      #=============================
      # añadir en memoria
      #==============================
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
                          # popleft si se alcanza la MAX_MEMORIA
     
        #=============================================
        # Entrenar memoria de largo plazo
        #===================================
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #============================================================
        # Entrenar memoria a corto plazo
        #============================================================

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        #==================================
        # Decidir acción
        #==================================

    def get_action(self, state):
        #=======================================================================
        # movimientos al azar:  balance entre exploración / explotación
        # JUEGOS_AL_AZAR  juegos con posibilidad de hacer un movimiento al azar
        #==========================================================================
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
          #================================
          # Genera entero al azar ente 0 y 2
          #================================
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #=======================================
            # Dado state (R11) genera prediction (R3)
            #======================================
            prediction = self.model(state0)
            #===================================================
            # move es entero entre 0 y 2
            # es la entrada con valor máximo en prediction (R3)
            #=================================================
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        #================================================
        # Desición es un vector en R3 de con 0 o 1
        #=============================================
        return final_move
#================================================
# FUNCIÓN PRINCIPAL : ENTRENAMIENTO
#============================================

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    #==============================
    # Ciclo infinito
    #============================
    while True:
        #========================
        # Obtener estado anterior
        #=======================
        state_old = agent.get_state(game)

        #============================
        # obtener movimiento
        #====================
        final_move = agent.get_action(state_old)

        #============================
        # Mover y obtener nuevo estado
        #==============================
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #=========================
        # Entrenar memoria corta
        #========================
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #============
        # Recordar
        #============
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #===========================================
            # Entrenar memoria larga , graficar resultado
            #=============================================
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

# PROGRAMA PRINCIPAL
if __name__ == '__main__':
    train()
