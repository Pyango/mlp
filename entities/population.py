import random
import time
from multiprocessing import Pool

from entities.genome import Genome
from entities.specie import Specie, DistanceCache


class Population:
    last_species_count = 0

    def __init__(
            self,
            num_inputs,
            num_outputs,
            fitness_threshold,
            initial_fitness,
            output_activation_functions,
            **kwargs
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness_threshold = fitness_threshold
        self.initial_fitness = initial_fitness
        self.output_activation_functions = output_activation_functions
        self.size = kwargs.get('size', 100)
        self.compatibility_threshold = kwargs.get('compatibility_threshold', 3)
        self.survival_threshold = kwargs.get('survival_threshold', 3)
        self.max_species = kwargs.get('survival_threshold', 10)
        self.compatibility_threshold_mutate_power = kwargs.get('survival_threshold', 0.01)
        self.generation = kwargs.get('initial_generation', 0)

        # Structure
        self.genomes = {}
        self.species = {}

        for i in range(self.size):
            self.create_genome()

    def create_genome(self):
        genome = Genome(
            key=self.get_new_genome_key(),
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            initial_fitness=self.initial_fitness,
            output_activation_functions=self.output_activation_functions,
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

    def run(self, compute_fitness, on_success, on_generation=None, generations=100):
        for generation in range(generations):

            """
            Print section
            """

            print('')
            print('#######################')
            print('')
            print(f'Current Generation: {generation}')
            print(f'Number of Species {len(self.species)}')
            print(f'Number of Genomes {len(self.genomes)}')
            print(f'Compatibility threshold {self.compatibility_threshold}')

            # Execute the custom implemented fitness function from the developer
            start_time = time.time()
            with Pool() as p:
                results = p.map(compute_fitness, self.genomes.values())
                self.genomes = {g.key: g for g in results}
            print(f"--- {time.time() - start_time} seconds for compute fitness ---")

            start_time = time.time()
            # Define the best genome
            best = None
            worst = None
            for g in self.genomes.values():
                if best is None or g.fitness > best.fitness:
                    best = g
                if worst is None or g.fitness < worst.fitness:
                    worst = g

            if on_generation:
                on_generation(best, population=self)

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

            # Some printing
            print(f'And the best genome is: {best.key} with a fitness of {best.fitness}'
                  f' and a complexity of {best.complexity} and adj fitness {best.adjusted_fitness}')
            if best.ancestors:
                print(f'The ancestors of the best genome are', best.ancestors[0].key, best.ancestors[1].key)
            print(f'And the worst genome is: {worst.key} with a fitness of {worst.fitness}'
                  f' and a complexity of {worst.complexity} and adj fitness {worst.adjusted_fitness}')

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

            """
            Kill stagnated genomes
            """
            for genome in [g for g in self.genomes.values()]:
                if genome.generation > self.survival_threshold and genome.last_fitness >= genome.fitness:
                    del self.genomes[genome.key]
                    self.create_genome()

            for genome in self.genomes.values():
                genome.last_fitness = genome.fitness

            print(f"--- {time.time() - start_time} to eval the species and mutate ---")
        if on_success:
            on_success(best)
        best.show()
