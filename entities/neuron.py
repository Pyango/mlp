from entities.activation import sigmoid
from entities.attribute import Attribute


class Neuron:
    def __init__(self, key, bias=1.0, activation_function=sigmoid):
        self.key = key
        self.bias = Attribute(
            value=bias,
            max_value=30,
            min_value=-30,
        )
        self.activation_function = activation_function
        self.value = 1
        self.connections = {}

    def activate(self):
        results = []
        for c in self.connections.values():
            results.append(c.input_neurone.value * c.weight * self.bias)
        return self.activation_function(
            sum(results)
        )

    def distance(self, other_neuron):
        """
        The genetic distance from this neuron to the other based on the bias and value (response)
        """
        # TODO: Update to compute differences when dealing with different activation functions as well
        return abs(self.bias - other_neuron.bias) + abs(self.value - other_neuron.value)
