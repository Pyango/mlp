from entities.activation import sigmoid
from entities.attribute import Attribute


class Neuron:
    def __init__(self, key, bias=1.0, activation_function=sigmoid):
        self.key = key
        self.bias = Attribute(
            value=bias,
            max_value=30,
            min_value=-30,
            mutate_rate=0.7,
            replace_rate=0.1,
            mutate_power=0.5,
        )
        self.activation_function = activation_function
        self.value = 1
        self.connections = {}

    def __del__(self):
        for connection in self.connections:
            del connection

    def activate(self):
        results = []
        for c in self.connections.values():
            results.append((c.input_neurone.value + self.bias) * c.weight)
        result = self.activation_function(
            sum(results)
        )
        return result

    def distance(self, other_neuron):
        """
        The genetic distance from this neuron to the other based on the bias and value (response)
        """
        # TODO: Update to compute differences when dealing with different activation functions as well
        return abs(self.bias - other_neuron.bias) + abs(self.value - other_neuron.value)

    def mutate(self):
        self.bias.mutate_value()
