import numpy as np
from tools import JsonProcessor
from model.Activation import Activation
from model.Loss import Loss
from model.Optimizer import Optimizer


class NeuralNetwork:
    """Neural Network model
    """

    def __init__(
            self,
            input_size,
            output_size,
            output_activation,
            hidden_layer_nodes=None,
            hidden_layer_activations=None,
            loss='mse',
            optimizer='sgd'
    ):
        """NeuralNetwork constructor

        :param input_size: number of inputs
        :type input_size: int

        :param output_size: number of outputs
        :type output_size: int

        :param output_activation: activation function of the output
        :type output_activation: str


        :param hidden_layer_nodes: number of nodes per layer in the hidden layer
        :type hidden_layer_nodes: list

        :param hidden_layer_activations: the activation function per layer in the hidden layer
        :type hidden_layer_activations: list

        :param loss: loss function for determining the correctness of the output (default: mse)
        :type loss: str

        :param optimizer: optimization algorithm used to update the parameters (default: sgd)
        :type optimizer: str
        """

        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_layer_nodes = hidden_layer_nodes
        self.hidden_layer_activations = hidden_layer_activations
        self.loss = loss

        # The number of layers in the neural network
        self.network_size = len(self.hidden_layer_activations) + 1

        # Parameters
        self.weights = []
        self.biases = []

        # Output of forward propagation
        self.outputs = []

        # Output of backward propagation
        self.deltas = []

        # Gradients from backpropagation
        self.grads = []

        # The optimizer of the network which lessens the error of the parameters
        optimizer_object = Optimizer()
        self.optimizer = getattr(optimizer_object, optimizer)

        # Dataset
        self.x = np.array([])
        self.y = np.array([])

    def load_data(self, file):
        """Gets the data from an external file
        """

        dataset = JsonProcessor.load(file)

        self.x = np.asarray(dataset['input'])
        self.y = np.asarray(dataset['output'])

    def train(self, epochs=1):
        """Trains the Neural Network

        :param epochs: The number of times optimizing the network
        :type epochs: int
        """

        for epoch in range(epochs):
            step = 0
            for i, o in zip(self.x, self.y):
                prediction = self.forward_pass(i)
                error = getattr(Loss, self.loss)(prediction, o)

                print('Error at epoch %d step %d: %f' % (epoch, step, error))

                self.backward_pass(prediction, o, i)
                self.parameter_update()

                step += 1

    def predict(self, inputs):
        """Generates a prediction using the inputs

        :param inputs: The input from the dataset
        :type inputs: numpy.ndarray|list

        :rtype: numpy.ndarray
        """

        # Make sure the input will be a numpy.ndarray
        inputs = np.asarray(inputs)

        return np.asarray(list(map(self.forward_pass, inputs)))

    def set_parameters(self):
        """Initializes the parameters, weights and biases
        """

        size_of_prev_layer = self.input_size

        # We first make the parameters for the hidden layer
        for size_of_cur_layer in self.hidden_layer_nodes:
            weight = np.random.random((size_of_prev_layer, size_of_cur_layer))
            bias = np.random.random(size_of_cur_layer)

            self.weights.append(weight)
            self.biases.append(bias)

            size_of_prev_layer = size_of_cur_layer

        # Then we make one for the output layer
        weight = np.random.random((size_of_prev_layer, self.output_size))
        bias = np.random.random(self.output_size)

        self.weights.append(weight)
        self.biases.append(bias)

    def forward_pass(self, inputs):
        """Does the forward propagation of the neural network to get its prediction

        :param inputs: The input from the dataset
        :type inputs: numpy.ndarray

        :rtype: numpy.ndarray
        """

        for index in range(self.network_size):
            # Compute the dot product of the input with their corresponding weights and then add the bias
            layer_sum = np.sum(inputs[:, None] * self.weights[index], axis=0) + self.biases[index]

            # Do note that the expression below is just the same as above
            # layer_sum = np.dot(self.weights[index], inputs) + self.biases[index]

            # We then apply an activation function to the neural network
            # But before that, we need to know if we're now at the output layer
            # so that we can use the right activation function
            activation = self.output_activation \
                if index == self.network_size - 1 \
                else self.hidden_layer_activations[index]

            layer_output = getattr(Activation, activation)(layer_sum)

            # Now that we have this layer's output, it's time to make this as input for the next layer
            inputs = layer_output

            # Let's also have this output store to be used later in backpropagation
            self.outputs.append(layer_output)

        return self.outputs[-1]

    def backward_pass(self, prediction, real, inputs):
        """Does the backpropagation, used to find the gradients for updating the parameters

        :param prediction: The output of the neural network
        :type prediction: numpy.ndarray

        :param real: The ground truth from the dataset
        :type real: numpy.ndarray

        :param inputs: The inputs from the dataset
        :type inputs: numpy.ndarray
        """

        # We first get the partial derivative of the error with respect to the predictions
        pd_loss_wrt_layer_output = getattr(Loss, 'pd_%s' % self.loss)(prediction, real)

        # Then we get the delta of the output layer
        delta = pd_loss_wrt_layer_output * getattr(Activation, 'pd_%s' % self.output_activation)(self.outputs[-1])

        # When then the delta when updating the output layer's bias
        self.deltas.insert(0, delta)

        # Since propagation goes back, we have to also iterate backwards
        for index in range(self.network_size - 2, -1, -1):
            # We then get the gradient for the parameters in the hidden layer
            grad = delta * self.outputs[index][:, None]

            # Store that so we can use it when optimizing the network's parameters
            self.grads.insert(0, grad)

            # We then get the partial derivative of the error with respect to the output of the hidden layers
            pd_loss_wrt_layer_output = np.sum(delta * self.weights[index + 1], axis=1)

            activation = self.hidden_layer_activations[index]

            delta = pd_loss_wrt_layer_output * getattr(Activation, 'pd_%s' % activation)(self.outputs[index])
            self.deltas.insert(0, delta)

        # At the end, we solve the gradients by multiplying the delta with its inputs
        grad = delta * inputs[:, None]

        self.grads.insert(0, grad)

        # Lastly, we need to reset the outputs
        self.outputs = []

    def parameter_update(self):
        """Updates the weights and biases of the neural network
        """

        for index in range(self.network_size):
            # We update the weights by passing it along with the gradients
            self.weights[index] = self.optimizer(self.weights[index], self.grads[index])

            # As for the biases, we pass them along with the deltas
            self.biases[index] = self.optimizer(self.biases[index], self.deltas[index])

        # Like the outputs, deltas and gradients needs to reset
        self.deltas = []
        self.grads = []
