from entities.attribute import Attribute


class Connection:
    def __init__(self, key, input_neurone, output_neurone, weight=0.0):
        self.key = key
        self.input_neurone = input_neurone
        self.output_neurone = output_neurone
        self.weight = Attribute(
            value=weight,
            max_value=30,
            min_value=-30,
        )

    def mutate(self):
        self.weight.mutate_value()
