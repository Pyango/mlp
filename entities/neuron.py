from random import random

from entities.activation import sigmoid, relu
from entities.attribute import Attribute


class Neuron:
    def __init__(self, key, bias=1, activation_function=relu):
        self.key = key
        self.bias = Attribute(
            value=bias,
            max_value=30,
            min_value=-30,
            mutate_rate=0.6,
            mutate_power=0.8,
            replace_rate=0.1,
        )
        self.activation_function = activation_function
        self.value = 0
        self.connections = {}

    def __repr__(self):
        return f"""Neuron(Key: {self.key}, Bias: {self.bias})"""

    def __del__(self):
        for connection in self.connections:
            del connection

    def activate(self):
        results = [self.bias]
        for c in self.connections.values():
            results.append(c.input_neurone.value * c.weight)
        self.value = self.activation_function(sum(results))
        return self.value

    def distance(self, other_neuron):
        """
        The genetic distance from this neuron to the other based on the bias and value (response)
        """
        # TODO: Update to compute differences when dealing with different activation functions as well
        return abs(self.bias - other_neuron.bias) + abs(self.value - other_neuron.value)

    def mutate(self):
        self.bias.mutate_value()

    def copy(self):
        return self.__class__(
            key=self.key,
            bias=self.bias.value,
            activation_function=self.activation_function,
        )

    def crossover(self, neurone1):
        """
        Make a new neurone and take the bias from this or the other genome
        """

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        return self.__class__(
            key=self.key,
            bias=neurone1.bias.value if random() > 0.5 else self.bias.value,
            activation_function=self.activation_function,
        )
