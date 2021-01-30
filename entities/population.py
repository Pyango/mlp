from entities.genome import Genome
from entities.specie import Specie, DistanceCache

xor2 = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
]

xor3 = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
]

xand = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
]

nand = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1],
]


class Population:
    compatibility_threshold = 2

    def __init__(self, num_inputs, num_outputs, initial_fitness, fitness_threshold, size=100):
        self.size = size
        self.initial_fitness = initial_fitness
        self.fitness_threshold = fitness_threshold
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
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

    def run(self, compute_fitness, generations=3000):
        for generation in range(generations):
            # Execute the custom implemented fitness function from the developer
            compute_fitness(self.genomes.items())

            best = None
            for g in self.genomes.values():
                if best is None or g.fitness > best.fitness:
                    best = g
            print(f'Generation: {generation}')
            print(f'And the best genome is: {best.key} with a fitness of {best.fitness}'
                  f' and a complexity of {best.complexity} and adj fitness {best.adjusted_fitness}')

            best.show()
            print(best)
            for i in xor2:
                print(f'{i} -> {best.activate(i[:2])}')

            if best.fitness >= self.fitness_threshold:
                break

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

            # Compute adjusted fitness for each genome in each specie
            for specie in self.species.values():
                for genome in specie.genomes.values():
                    genome.adjusted_fitness = genome.fitness / len(specie.genomes)

            species_avg_fitness = [s.avg_fitness for s in self.species.values()]
            print(f'Number of species {len(self.species)}')

            # Mutate the best 10% - 20% of all genomes
            genomes = sorted([g for g in self.genomes.values()], reverse=True)
            top_genomes = genomes[int(len(genomes) * .2): int(len(genomes) * .3)]
            for g in top_genomes:
                g.mutate()

            # Mutate the worst 10% genomes
            genomes = sorted([g for g in self.genomes.values()], reverse=False)
            bad_genomes = genomes[int(len(genomes) * .0): int(len(genomes) * .10)]
            for g in bad_genomes:
                g.mutate()
            print()
