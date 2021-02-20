import gym
import numpy as np

from entities.activation import clamped_activation
from entities.population import Population

population = Population(
    num_inputs=24,
    num_outputs=4,
    fitness_threshold=200,  # frames how long it should survive
    initial_fitness=0,
    survival_threshold=0,
    compatibility_threshold=1,
    max_species=2,
    size=20,
    output_activation_function=clamped_activation,
)

runs_per_net = 1


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        fitnesses = []
        for runs in range(runs_per_net):
            env = gym.make("BipedalWalker-v3")
            observation = env.reset()
            fitness = 0.0
            done = False
            while not done:
                action = genome.activate(observation)
                observation, reward, done, info = env.step(action)
                fitness += reward
            fitnesses.append(fitness)
        genome.fitness = np.mean(fitnesses)


def on_success(best):
    env = gym.make("BipedalWalker-v3")
    observation = env.reset()
    done = False
    while not done:
        action = best.activate(observation)
        observation, reward, done, info = env.step(action)
        env.render()


population.run(compute_fitness, on_success, generations=300)
