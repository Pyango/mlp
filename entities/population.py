import random

from entities.genome import Genome
from entities.specie import Specie, DistanceCache


class Population:
    last_species_count = 0

    def __init__(self, num_inputs, num_outputs, initial_fitness, fitness_threshold, output_activation_function,
                 size=100, compatibility_threshold=3, survival_threshold=0, max_species=30,
                 compatibility_threshold_mutate_power=.01):
        self.size = size
        self.initial_fitness = initial_fitness
        self.fitness_threshold = fitness_threshold
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genomes = {}
        self.species = {}
        self.compatibility_threshold = compatibility_threshold
        self.compatibility_threshold_mutate_power = compatibility_threshold_mutate_power
        self.survival_threshold = survival_threshold
        self.max_species = max_species
        self.output_activation_function = output_activation_function
        for i in range(self.size):
            self.create_genome()

    def create_genome(self):
        genome = Genome(
            key=self.get_new_genome_key(),
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            initial_fitness=self.initial_fitness,
            output_activation_function=self.output_activation_function,
        )
        self.genomes[genome.key] = genome
        return genome

    def get_new_genome_key(self):
        if self.genomes:
            return max(self.genomes.keys()) + 1
        return 1

    def get_new_specie_key(self):
        if self.species:
            return max(self.species.keys()) + 1
        return 1

    def run(self, compute_fitness, on_success, generations=100):
        for generation in range(generations):
            # Execute the custom implemented fitness function from the developer
            compute_fitness(self.genomes.items())

            # Define the best genome
            best = None
            for g in self.genomes.values():
                if best is None or g.fitness > best.fitness:
                    best = g

            # Some printing
            print(f'Generation: {generation}')
            print(f'And the best genome is: {best.key} with a fitness of {best.fitness}'
                  f' and a complexity of {best.complexity} and adj fitness {best.adjusted_fitness}')
            if best.ancestors:
                print(f'The ancestors of the best genome are', best.ancestors[0].key, best.ancestors[1].key)

            if best.fitness >= self.fitness_threshold:
                break

            # Calculate the distances for speciation
            distances = DistanceCache()
            self.species = {}
            genome_to_species = {}
            for g in self.genomes.values():
                g.generation += 1
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
                    if og.key not in genome_to_species:
                        distance = distances(g, og)
                        if distance <= self.compatibility_threshold:
                            specie.genomes[og.key] = og
                            genome_to_species[og.key] = specie.key

            if len(self.species) > self.max_species:
                self.compatibility_threshold += (len(
                    self.species) - self.max_species) * self.compatibility_threshold_mutate_power
            self.last_species_count = len(self.species)

            # Compute adjusted fitness for each genome in each specie
            for specie in self.species.values():
                for genome in specie.genomes.values():
                    genome.adjusted_fitness = genome.fitness / len(specie.genomes)
                    if genome == best:
                        print(f'Best {genome.key} is in specie {specie.key} with {len(specie.genomes)} members.')

            """
            Print section
            """

            print(f'Species {len(self.species)}')
            print(f'Genomes {len(self.genomes)}')
            print(f'Compatibility threshold {self.compatibility_threshold}')

            """
            Crossover and mutation
            """
            top_genomes = sorted(
                [g for g in self.genomes.values()],
                reverse=True,
            )
            bad_genomes = sorted(
                [g for g in self.genomes.values()],
                reverse=False,
            )

            # Mutate the best 20% - 40% of all genomes
            for g in top_genomes[int(len(top_genomes) * .2): int(len(top_genomes) * .4)]:
                g.mutate()

            # Crossover the best 0% - 10% of all genomes and delete the same amount of the worst ones
            crossover_genomes = top_genomes[int(len(top_genomes) * .0): int(len(top_genomes) * .1)]
            genomes_to_delete = bad_genomes[int(len(bad_genomes) * .1): int(len(bad_genomes) * .2)]

            for bad_genome in genomes_to_delete:
                parent1 = random.choice(crossover_genomes)
                parent2 = random.choice(crossover_genomes)
                new_genome = self.create_genome()
                new_genome.crossover(parent1, parent2)
                new_genome.mutate()
                del self.genomes[bad_genome.key]

            # Delete the worst 10% genomes and let them rebirth: stagnation mechanism
            for g in bad_genomes[int(len(bad_genomes) * .0): int(len(bad_genomes) * .1)]:
                if g.key in self.genomes:
                    if self.genomes[g.key].generation > self.survival_threshold:
                        del self.genomes[g.key]
                        self.create_genome()
                    else:
                        g.mutate()
        if on_success:
            on_success(best)
        best.show()
