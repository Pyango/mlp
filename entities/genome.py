from random import random, choice

from graphviz import Digraph

from entities.connection import Connection
from entities.neuron import Neuron


class Genome:
    # Static config variables go here
    compatibility_disjoint_coefficient = 1
    adjusted_fitness = None
    neuron_add_prob = .05
    neuron_delete_prob = .001
    conn_add_prob = .2
    conn_delete_prob = .1
    generation = 0

    def __repr__(self):
        return f"Genome {self.key}\n## Neurones\n{self.neurones}\n## Connections\n{self.connections})"

    def __lt__(self, other):
        return self.adjusted_fitness < other

    def __gt__(self, other):
        return self.adjusted_fitness > other

    def __ge__(self, other):
        return self.adjusted_fitness >= other

    def __le__(self, other):
        return self.adjusted_fitness <= other

    def __init__(self, key, num_inputs, num_outputs, initial_fitness):
        self.key = key
        self.connections = {}
        self.input_neurones = {}
        self.output_neurones = {}
        self.hidden_neurones = {}
        self.fitness = initial_fitness
        # Input neurons have negative keys
        for i in range(num_inputs):
            n = Neuron(
                key=-i - 1,
            )
            self.input_neurones[n.key] = n

        # Output neurons have positive keys
        for i in range(num_outputs):
            n = Neuron(
                key=i + 1,
            )
            self.output_neurones[n.key] = n

        # Fully connect all input to all output neurons
        for input_neurone in self.input_neurones.values():
            for output_neurone in self.output_neurones.values():
                connection = Connection(
                    input_neurone=input_neurone,
                    output_neurone=output_neurone,
                )
                output_neurone.connections[connection.key] = connection
                self.connections[connection.key] = connection

    def show(self):
        dot = Digraph(format='png')
        for n in self.input_neurones.values():
            dot.node(str(n.key), f'In {round(n.value, 2)}\n{round(n.bias, 2)}')
        for n in self.output_neurones.values():
            dot.node(str(n.key), f'Out {round(n.value, 2)}\n{round(n.bias, 2)}')
        for n in self.hidden_neurones.values():
            dot.node(str(n.key), f'{round(n.value, 2)}\n{round(n.bias, 2)}')
        for key, c in self.connections.items():
            dot.edge(str(c.input_neurone.key), str(c.output_neurone.key), label=f'{round(c.weight, 2)}')
        dot.render(f'./network-{self.key}', view=False)

    @property
    def neurones(self):
        return {**self.hidden_neurones, **self.input_neurones, **self.output_neurones}

    def activate(self, inputs):
        # Set all the values of the input neurones
        for input_value, input_neurone in zip(inputs, self.input_neurones.values()):
            input_neurone.value = input_value

        for n in self.hidden_neurones.values():
            n.activate()

        outputs = []
        for output_neurone in self.output_neurones.values():
            outputs.append(output_neurone.activate())
        return outputs

    def get_new_neurone_key(self):
        if self.neurones:
            return max(self.neurones.keys()) + 1
        return 1

    def get_new_connection_key(self):
        return max(self.connections.keys()) + 1

    def create_neuron(self):
        neurone = Neuron(
            key=self.get_new_neurone_key(),
        )
        self.hidden_neurones[neurone.key] = neurone
        return neurone

    def create_connection(self, input_neurone, output_neurone, **kwargs):
        c = Connection(
            input_neurone=input_neurone,
            output_neurone=output_neurone,
            **kwargs,
        )
        self.connections[c.key] = c
        output_neurone.connections[c.key] = c
        return c

    def mutate_add_neurone(self):
        if not self.connections:
            self.mutate_add_connection()
            return
        connection_to_split = choice(list(self.connections.values()))
        new_neuron = self.create_neuron()
        self.create_connection(
            input_neurone=connection_to_split.input_neurone,
            output_neurone=new_neuron,
            weight=connection_to_split.weight.value,
        )
        self.create_connection(
            input_neurone=new_neuron,
            output_neurone=connection_to_split.output_neurone,
            weight=connection_to_split.weight.value,
        )
        # Delete old connection
        del self.connections[connection_to_split.key]

    def mutate_delete_neurone(self):
        if self.hidden_neurones:
            del_key = choice(list(self.hidden_neurones.keys()))
            del self.hidden_neurones[del_key]

    def mutate_add_connection(self):
        """
        Attempt to add a new connection, the only restriction being that the output
        neurone cannot be one of the network input neurones.
        """

        possible_inputs = list(self.input_neurones.values()) + list(self.hidden_neurones.values())
        input_neurone = choice(possible_inputs)

        possible_outputs = list(self.hidden_neurones.values()) + list(self.output_neurones.values())
        output_neurone = choice(possible_outputs)

        # Don't allow connections to the same neurone because its not supported yet
        if input_neurone.key == output_neurone.key:
            return
        # Don't duplicate connections and avoid same neurone connections
        if (input_neurone.key, output_neurone.key) in self.connections.keys():
            return
        self.create_connection(input_neurone, output_neurone)

    def mutate_delete_connection(self):
        if self.connections:
            del_key = choice(list(self.connections.keys()))
            del self.connections[del_key]

    def mutate(self):
        """ Mutates this genome. """

        if random() < self.neuron_add_prob:
            self.mutate_add_neurone()

        if random() < self.neuron_delete_prob:
            self.mutate_delete_neurone()

        if random() < self.conn_add_prob:
            self.mutate_add_connection()

        if random() < self.conn_delete_prob:
            self.mutate_delete_connection()

        # Mutate connection genes.
        for connection in self.connections.values():
            connection.mutate()

        # Mutate neurones genes.
        for neurone in self.neurones.values():
            neurone.mutate()

    def distance(self, other):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
        # TODO: Is there a better way to compute the distance between 2 neurones?
        # TODO: Cleanup this mess!

        neuron_distance = 0.0
        if self.neurones or other.neurones:
            disjoint_neurones = 0
            for k2 in other.neurones:
                if k2 not in self.neurones:
                    disjoint_neurones += 1

            for k1, n1 in self.neurones.items():
                n2 = other.neurones.get(k1)
                if n2 is None:
                    disjoint_neurones += 1
                else:
                    # Homologous genes compute their own distance value.
                    neuron_distance += n1.distance(n2)

            max_neurones = max(len(self.neurones), len(other.neurones))
            neuron_distance = (neuron_distance + (
                    self.compatibility_disjoint_coefficient * disjoint_neurones)
                               ) / max_neurones

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (self.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = neuron_distance + connection_distance
        return distance

    @property
    def complexity(self):
        """
        Returns genome 'complexity', taken to be
        (number of neurones, number of connections)
        """
        return len(self.neurones), len(self.connections)

    def crossover(self, genome1, genome2):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        for key, neurone1 in parent1.neurones.items():
            neurone2 = parent2.neurones.get(key)
            assert key not in self.neurones
            if neurone2 is None:
                # Extra gene: copy from the fittest parent
                self.neurones[key] = neurone1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.neurones[key] = neurone1.crossover(neurone2)
