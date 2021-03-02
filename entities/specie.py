class DistanceCache:
    def __init__(self):
        self.distances = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        gk0 = genome0.key
        gk1 = genome1.key
        d = self.distances.get((gk0, gk1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1)
            self.distances[gk0, gk1] = d
            self.distances[gk1, gk0] = d
            self.misses += 1
        else:
            self.hits += 1
        return d


class Specie:
    def __init__(self, key, genomes):
        self.key = key
        self.genomes = genomes

    @property
    def avg_fitness(self):
        combined_fitness = sum([g.fitness for g in self.genomes.values()])
        return combined_fitness / len(self.genomes)
