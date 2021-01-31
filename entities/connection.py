from entities.attribute import Attribute


class Connection:
    compatibility_weight_coefficient = .5

    def __init__(self, input_neurone, output_neurone, weight=1.0):
        self.input_neurone = input_neurone
        self.output_neurone = output_neurone
        self.key = (input_neurone.key, output_neurone.key)
        self.weight = Attribute(
            value=weight,
            max_value=30,
            min_value=-30,
            mutate_rate=0.6,
            mutate_power=0.5,
            replace_rate=0.1,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'{self.input_neurone.key} -> {self.output_neurone.key} | {self.weight.value}'

    def mutate(self):
        self.weight.mutate_value()

    def distance(self, other):
        d = abs(self.weight - other.weight)
        return d * self.compatibility_weight_coefficient

    def copy(self, input_neurone, output_neurone):
        return self.__class__(
            input_neurone=input_neurone,
            output_neurone=output_neurone,
            weight=self.weight.value,
        )
