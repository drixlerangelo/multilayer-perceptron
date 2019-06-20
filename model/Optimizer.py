import numpy
from tools import JsonProcessor


class Optimizer:
    """Optimization algorithms to minimize (or maximize) the neural network's loss
    """

    def __init__(self):
        """Optimizer constructor
        """

        self.constants = JsonProcessor.load('optimizer_constants.json')

    def sgd(self, parameter, gradient):
        """Stochastic Gradient Descent

        :param parameter:
        :type parameter: numpy.ndarray

        :param gradient:
        :type gradient: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return parameter - (self.constants['eta'] * gradient)
