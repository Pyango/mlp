from itertools import groupby

from entities.genome import Genome
from entities.specie import Specie, DistanceCache


class Population:
    compatibility_threshold = .1

    def __init__(self, num_inputs, num_outputs, initial_fitness, fitness_threshold, size=100):
        self.size = size
        self.initial_fitness = initial_fitness
        self.fitness_threshold = fitness_threshold
        self.genomes = {}
        self.species = {}
        for i in range(self.size):
            genome = Genome(
                key=self.get_new_genome_key(),
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                initial_fitness=initial_fitness
            )
            self.genomes[genome.key] = genome

    def get_new_genome_key(self):
        if self.genomes:
            return max(self.genomes.keys()) + 1
        return 1

    def get_new_specie_key(self):
        if self.species:
            return max(self.species.keys()) + 1
        return 1

    def run(self, compute_fitness, generations=300):
        for generation in range(generations):
            compute_fitness(self.genomes.items())

            best = None
            for g in self.genomes.values():
                if best is None or g.fitness > best.fitness:
                    best = g

            print(f'And the best genome is: {best.key} with a fitness of {best.fitness}'
                  f' and a complexity of {best.complexity}')

            distances = DistanceCache()
            self.species = {}
            for g in self.genomes.values():
                sk = self.get_new_specie_key()
                self.species[sk] = Specie({
                    g.key: g,
                })
                for og in self.genomes.values():
                    distance = distances(g, og)
                    if distance <= self.compatibility_threshold:
                        self.species[sk].genomes[og.key] = og
                # here
                if len(self.species[sk].genomes) <= 1:
                    del self.species[sk]

            # species = {}
            # for g in self.genomes.values():
            #     for og in self.genomes.values():
            #         distance = distances(g, og)
            #         if distance > self.compatibility_threshold:
            #             continue
            #         for s in species.values():
            #             if g.key in s.genomes:
            #                 s.genomes[og.key] = og
            #             elif og.key in s.genomes:
            #                 s.genomes[g.key] = g
            #         else:
            #             sk = self.get_new_specie_key()
            #             species[sk] = Specie({
            #                 g.key: g,
            #                 og.key: og,
            #             })
            #     print()  # put a breakpoint here, i can't do it
            # How to make species out of them?
            # for key, group in groupby(map(lambda x: Distance(*x), distances)):
            #     genomes = {el.genome1.key: el.genome1 for el in group}
            #     species.append(Specie(genomes))
            print()
