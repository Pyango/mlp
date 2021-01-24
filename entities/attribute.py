from random import random, gauss


class Attribute:

    def __init__(self, value, max_value, min_value, mutate_rate=.1, mutate_power=.1, replace_rate=.01):
        self.value = value
        self.initial_value = value
        self.max_value = max_value
        self.min_value = min_value
        self.mutate_rate = mutate_rate
        self.mutate_power = mutate_power
        self.replace_rate = replace_rate

    def __float__(self):
        return self.value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __abs__(self):
        return self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __str__(self):
        return str(self.value)

    def clamp(self, value):
        return max(min(value, self.max_value), self.min_value)

    def mutate_value(self):

        r = random()

        if r < self.mutate_rate:
            self.value = self.clamp(self.value + gauss(0.0, self.mutate_power))
            return self.value

        if r < self.replace_rate + self.mutate_rate:
            self.value = self.initial_value
            return self.value
