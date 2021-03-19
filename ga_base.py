import numpy as np
from multiprocessing import Pool

class GeneticAlgorithmBase:

    def __init__(self, size, survivial_rate=0.2, mutation_rate=0.1, n_iter=1000):

        self.size = size
        self.survivial_rate = survivial_rate
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        assert isinstance(size, int) and size >= 2, "size must be integer and greater or equal than two"
        self._size = size

    @property
    def survivial_rate(self):
        return self._survivial_rate

    @survivial_rate.setter
    def survivial_rate(self, rate):
        assert 0 <= rate <= 1, "survivial_rate must be a probability"
        self._survivial_rate = rate

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, rate):
        assert 0 <= rate <= 1, "mutation_rate must be a probability"
        self._mutation_rate = rate

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, n):
        assert isinstance(n, int) and n >= 0, "n_iter must be integer and at least one"
        self._n_iter = n

    def _fitness(self, x):
        raise NotImplementedError

    def _crossover(self):
        raise NotImplementedError

    def _mutation(self):
        raise NotImplementedError

    def _selection(self, population, multiprocessing=False):
        if multiprocessing:
            with Pool(None) as p:
                fit_map = np.array(p.map(self._fitness, population))
        else:
            fit_map = np.array([self._fitness(pop) for pop in population])
        fit_sum = fit_map.sum()
        p = [x/fit_sum for x in fit_map]
        k = max(2, self._survivial_rate*self._size)
        indices = np.random.choice(np.arange(self._size, p=p), k)

        return population[indices], fit_map[indices]

    def run(self, population, multiprocessing=False):

        highest_fitness_all_time = 0
        best_individual_all_time = None

        for _ in range(n_iter):
            parents, fitness_scores = self._selection(population, multiprocessing)
            offspring = self._mutation(self._crossover(parents))
            population = np.concatenate(parents, offspring)
        else:
            fit_map = [self._fitness(pop) for pop in population]
            population, fit_map = zip(*sorted(zip(population, fit_map), key=lambda x: x[1], reversed=True))

        return population[0], fit_map[0]
