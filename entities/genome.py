from random import random, choice

from entities.connection import Connection
from entities.neuron import Neuron


class Genome:
    def __init__(self, key, num_inputs, num_outputs):
        self.key = key
        self.connections = {}
        self.input_neurones = {}
        self.output_neurones = {}
        self.neurones = {}
        self.fitness = 0
        self.neurones_indexer = None
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
