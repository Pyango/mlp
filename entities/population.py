from entities.genome import Genome


class Population:
    def __init__(self, num_inputs, num_outputs, initial_fitness, fitness_threshold, size=100):
        self.size = size
        self.initial_fitness = initial_fitness
        self.fitness_threshold = fitness_threshold
        self.genomes = {}
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

    def run(self, compute_fitness, generations=300):
        for generation in range(generations):
            compute_fitness(self.genomes.items())

            best = None
            for g in self.genomes.values():
                if g.fitness is None:
                    raise RuntimeError(f"Fitness not assigned to genome {g.key}")

                if best is None or g.fitness > best.fitness:
                    best = g
            print(f'And the best genome is: {best.key} with a fitness of {best.fitness}'
                  f' and a complexity of {best.complexity}')

            distances = []
            for gk, g in self.genomes.items():
                for ogk, og in self.genomes.items():
                    if gk == ogk:
                        continue
                    distances.append((g, og, g.distance(og)))
            # How to make species out of them?
