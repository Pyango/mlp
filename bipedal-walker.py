import multiprocessing

import gym
import numpy as np
import pickle

from entities.activation import all_activation_functions
from entities.population import Population

population = Population(
    num_inputs=24,
    num_outputs=4,
    fitness_threshold=200,
    initial_fitness=0,
    survival_threshold=0,
    compatibility_threshold=1,
    max_species=10,
    size=30,
    output_activation_functions=all_activation_functions,
)

runs_per_net = 1


def compute_fitness(genome):
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
    return genome


def on_success(best):
    env = gym.make("BipedalWalker-v3")
    observation = env.reset()
    done = False
    while not done:
        action = best.activate(observation)
        observation, reward, done, info = env.step(action)
        env.render()
    outfile = open('best-bipeda-walker', 'wb')
    pickle.dump(best, outfile)
    outfile.close()


def run():
    multiprocessing.freeze_support()
    population.run(compute_fitness, on_success, generations=300)


if __name__ == '__main__':
    run()
