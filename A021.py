#Programa 21: Algoritmo genético simple
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
import datetime
import random

random.seed(random.random())
startTime = datetime.datetime.now()

#=================================
#  Los genes
#=================================
geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

#=================================
#  Objetivo
#=================================
target = "Hola mundo"

#=================================
#  Frase inicial
#=================================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sampleSize = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return "".join(genes)

#=================================
#  Función de aptitud
#=================================
def get_fitness(guess):
    return sum(1 for expected, actual in zip(target,guess) if expected == actual)

#=================================
#  Mutación de letras en la frase
#=================================
def mutate(parent):
    index = random.randrange(0,len(parent))
    childGenes[index] = alternate if newGene == childGenes[index] else newGene
    return "".join(childGene)

#=================================
#  Monitoreo de la solución
#=================================
def display(guess):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\t{}".format(guess,fitness,timeDiff))
    
#=================================
#  Codigo principal
#=================================
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParent)

#=================================
#  Iteraciones
#=================================
while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    if bestdFitness >= childFitness:
        display(child)
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bistFitness = childFitness
    bestParent = child
    












