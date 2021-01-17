from entities.genome import Genome

xor = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
]

new_genome = Genome(
    key=1,
    num_inputs=3,
    num_outputs=1,
)

while True:
    for i in xor:
        result = new_genome.activate(i[:3])
        new_genome.mutate()
        new_genome.fitness -= (result[0] - i[-1]) ** 2
        print(f'Fitness: {new_genome.fitness}')
