import math


def relu(z):
    """
    standard ReLu transfer/activation function
    returns z or 0 if z < 0
    included for testing
    """
    if z < 0:
        z = 0
    return z


def sigmoid(z):
    """
    modified sigmoidal transfer function suggested by the original NEAT
    """
    return 1 / (1 + math.e ** (-4.9 * z))
