from random import random, choice

from entities.connection import Connection
from entities.neuron import Neuron


class Genome:
    # Static config variables go here
    compatibility_disjoint_coefficient = 1

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
        counter = 1
        for input_neurone in self.input_neurones.values():
            for output_neurone in self.output_neurones.values():
                connection = Connection(
                    key=counter,
                    input_neurone=input_neurone,
                    output_neurone=output_neurone,
                )
                output_neurone.connections[connection.key] = connection
                self.connections[connection.key] = connection
                counter += 1

    @property
    def neurones(self):
        return self.hidden_neurones | self.input_neurones | self.output_neurones

    def activate(self, inputs):
        # Set all the values of the input neurones
        for input_value, input_neurone in zip(inputs, self.input_neurones.values()):
            input_neurone.value = input_value

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
        return Neuron(
            key=self.get_new_neurone_key(),
        )

    def create_connection(self, input_neurone, output_neurone, **kwargs):
        c = Connection(
            key=self.get_new_connection_key(),
            input_neurone=input_neurone,
            output_neurone=output_neurone,
            **kwargs,
        )
        self.connections[c.key] = c
        return c

    def mutate_add_node(self):
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

    def mutate(self):
        """ Mutates this genome. """

        if random() < .1:
            self.mutate_add_node()

        # Mutate connection genes.
        for connection in self.connections.values():
            connection.mutate()

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
