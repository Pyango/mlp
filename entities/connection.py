from random import random

from entities.attribute import Attribute


class Connection:
    compatibility_weight_coefficient = .5

    def __init__(self, input_key, output_key, weight=1.0):
        self.input_key = input_key
        self.output_key = output_key
        self.key = (self.input_key, self.output_key)
        self.weight = Attribute(
            value=weight,
            max_value=30,
            min_value=-30,
            mutate_rate=0.3,
            mutate_power=0.8,
            replace_rate=0.1,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{self.input_key} -> {self.output_key} | {self.weight.value}'

    def mutate(self):
        self.weight.mutate_value()

    def distance(self, other):
        d = abs(self.weight - other.weight)
        return d * self.compatibility_weight_coefficient

    def copy(self):
        return self.__class__(
            input_key=self.input_key,
            output_key=self.output_key,
            weight=self.weight.value,
        )

    def crossover(self, connection):
        """
        Make a new neurone and take the bias from this or the other genome
        """

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        return self.__class__(
            input_key=self.input_key,
            output_key=self.output_key,
            weight=connection.weight.value if random() > 0.5 else self.weight.value,
        )
