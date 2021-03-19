import numpy as np
from GeneticAlgorithmBase import GeneticAlgorithmBase

class GeneticAlgorithmVRP(GeneticAlgorithmBase):

    def _fitness(self, x):
        raise NotImplementedError

    def _crossover(self):
        raise NotImplementedError

    def _mutation(self):
        raise NotImplementedError
