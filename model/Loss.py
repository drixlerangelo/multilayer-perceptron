import numpy as np


class Loss:
    """Loss function used by the neural network
    """

    @staticmethod
    def mse(prediction, real):
        """Mean Squared Error

        :param prediction: output of the neural network
        :type prediction: numpy.ndarray

        :param real: the ground truth values
        :type real: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return np.sum((real - prediction) ** 2) / len(real)

    @staticmethod
    def pd_mse(prediction, real):
        """Partial derivative of Mean Squared Error

        :param prediction: output of the neural network
        :type prediction: numpy.ndarray

        :param real: the ground truth values
        :type real: numpy.ndarray

        :rtype: numpy.ndarray
        """

        return prediction - real
