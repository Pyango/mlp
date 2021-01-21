from entities.genome import Genome
from entities.specie import Specie, DistanceCache
from entities.utils import mean


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
            genome_to_species = {}
            for g in self.genomes.values():
                if g.key not in genome_to_species:
                    sk = self.get_new_specie_key()
                    specie = Specie(key=sk, genomes={
                        g.key: g,
                    })
                    self.species[sk] = specie
                    genome_to_species[g.key] = specie.key
                else:
                    specie = self.species[genome_to_species[g.key]]
                for og in self.genomes.values():
                    distance = distances(g, og)
                    if distance <= self.compatibility_threshold:
                        specie.genomes[og.key] = og
                        genome_to_species[og.key] = specie.key

            all_fitnesses = [g.fitness for g in self.genomes.values()]
            min_fitness = min(all_fitnesses)
            max_fitness = max(all_fitnesses)
            # Do not allow the fitness range to be zero, as we divide by it below.
            # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
            fitness_range = max(1.0, max_fitness - min_fitness)
            for afs in self.species.values():
                # Compute adjusted fitness.
                msf = mean([m.fitness for m in afs.genomes.values()])
                af = (msf - min_fitness) / fitness_range
                afs.adjusted_fitness = af
            adjusted_fitnesses = [s.adjusted_fitness for s in self.species.values()]
            avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
            print(f'Number of species {len(self.species)}, avg adj fitness {avg_adjusted_fitness}')
