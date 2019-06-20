import numpy as np


class Activation:
    """Activation functions used by the neural network
    """

    @staticmethod
    def sigmoid(inputs):
        """Sigmoid or Logistic

        :param inputs: The result of the dot product of the previous layer's output with their corresponding parameters
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return 1 / (1 + np.exp(-inputs))

    @staticmethod
    def pd_sigmoid(inputs):
        """Partial derivative of Sigmoid or Logistic

        :param inputs: The result of the parital derivative of the loss with respect the layer's output
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return inputs * (1 - inputs)

    @staticmethod
    def tanh(inputs):
        """Hyperbolic Tangent

        :param inputs: The result of the dot product of the previous layer's output with their corresponding parameters
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return (1 - np.exp(-2 * inputs)) / (1 + np.exp(-2 * inputs))

    @staticmethod
    def pd_tanh(inputs):
        """Partial derivative of Hyperbolic Tangent

        :param inputs: The result of the partial derivative of the loss with respect the layer's output
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return 1 - (inputs ** 2)

    @staticmethod
    def relu(inputs):
        """Rectified Linear Unit

        :param inputs: The result of the dot product of the previous layer's output with their corresponding parameters
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return np.where(inputs > 0, inputs, 0)

    @staticmethod
    def pd_relu(inputs):
        """Partial derivative of Rectified Linear Unit

        :param inputs: The result of the partial derivative of the loss with respect the layer's output
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return np.where(inputs > 0, 1, 0)

