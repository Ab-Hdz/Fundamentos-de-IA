#Programa 20: pso_clustering
#=================================
# Angelica Abigail Ramos Hernandez
# Fundamentos de IA
# Matematica Algoritmica
# ESFM IPN
# Junio 2025
#==========================

#=================================
#  Clase de optimización usando enjambres de particulas
#=================================

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from particle import Particle


class PSOClusteringSwarm:
    def __init__(self, n_clusters: int, n_particles: int, data: np.ndarray, hybrid=True, w=0.72, c1=1.49, c2=1.49):
        """
        Inicializa el enjambre
        :param n_clusters: numero de agrupamientos
        :param n_particles: numero de particulas
        :param data: (number_of_points x dimensiones)
        :param hybrid: bool : si hay que usar o no kmeans como semillero
        :param w:
        :param c1:
        :param c2:
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data

        self.particles = []
        # guardar el mejor en el enjambres
        self.gb_pos = None
        self.gb_val = np.inf
        # el mejor  agrupamiento hasta aqui
        # para cada dato contiene el número de agrupamiento
        self.gb_clustering = None

        self._generate_particles(hybrid, w, c1, c2)

    def _print_initial(self, iteration, plot):
        print('*** Initialing swarm with', self.n_particles, 'PARTICLES, ', self.n_clusters, 'CLUSTERS with', iteration,
              'MAX ITERATIONS and with PLOT =', plot, '***')
        print('Data=', self.data.shape[0], 'points in', self.data.shape[1], 'dimensions')

    def _generate_particles(self, hybrid: bool, w: float, c1: float, c2: float):
        """
        Genera particulas con k agrupamientos y puntos en t dimensiones
        :return:
        """
        for i in range(self.n_particles):
            particle = Particle(n_clusters=self.n_clusters, data=self.data, use_kmeans=hybrid, w=w, c1=c1, c2=c2)
            self.particles.append(particle)

    def update_gb(self, particle):
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.pb_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        """

        :param plot: = True graficara los  mejores agrupamientos globales
        :param iteration: numero de iteraciones maximas
        :return: (best cluster, best fitness value)
        """
        self._print_initial(iteration, plot)
        progress = []
        # Iterate until the max iteration
        for i in range(iteration):
            if i % 200 == 0:
                clusters = self.gb_clustering
                print('iteration', i, 'GB =', self.gb_val)
                print('best clusters so far = ', clusters)
                if plot:
                    centroids = self.gb_pos
                    if clusters is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=clusters, cmap='viridis')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:  # if there is no clusters yet ( iteration = 0 ) plot the data with no clusters
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_pb(data=self.data)
                self.update_gb(particle=particle)

            for particle in self.particles:
                particle.move_centroids(gb_pos=self.gb_pos)
            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])

        print('Finished!')
        return self.gb_clustering, self.gb_val