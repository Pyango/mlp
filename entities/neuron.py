from random import random, choice

from entities.activation import relu_activation, all_activation_functions
from entities.attribute import Attribute


class Neuron:
    activation_function_mutate_rate = 0.1

    def __init__(self, key, bias=1, output=False, activation_function=relu_activation):
        self.key = key
        self.value = 0
        self.bias = Attribute(
            value=bias,
            max_value=30,
            min_value=-30,
            mutate_rate=0.3,
            mutate_power=0.8,
            replace_rate=0.1,
        )
        self.activated = False
        self.output = output
        self.activation_function = activation_function

    def __repr__(self):
        return f"""Neuron(Key: {self.key}, Bias: {self.bias})"""

    def distance(self, other_neuron):
        """
        The genetic distance from this neuron to the other based on the bias and value (response)
        """
        # TODO: Update to compute differences when dealing with different activation functions as well
        return abs(self.bias - other_neuron.bias) + abs(self.value - other_neuron.value)

    def mutate(self):
        self.bias.mutate_value()

        r = random()
        if r < self.activation_function_mutate_rate:
            self.activation_function = choice(all_activation_functions)

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
