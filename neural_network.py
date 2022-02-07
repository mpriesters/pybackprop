"""
Neural Network with Backpropagation.

Author: Matthias Priesters
"""

from sklearn.metrics import mean_squared_error
import numpy as np
import numbers


class NeuralNetworkLayer:
    """NeuralNetworkLayer."""

    def __init__(self,
                 inputs,
                 nodes,
                 activation_function,
                 layer_num,
                 weights=None,
                 output_layer=False,
                 verbose=False):
        # initialize weights randomly between -0.01 and 0.01
        # if no weights are given
        if weights is None:
            self.weights = 0.02 * np.random.rand(inputs + 1, nodes) - 0.01
        else:
            self.weights = weights
        # static settings for this layer
        self.layer_num = layer_num
        self.output_layer = output_layer
        # get activation function and its derivative
        self.theta, self.theta_prime = get_activation_function(
            activation_function)

        # work variables for this layer
        self.input_values = None
        self.signal = None
        self.activation = None
        self.delta = None
        if verbose:
            print((f'initializing layer {layer_num} with {inputs} inputs, '
                   f'{nodes} nodes, function {activation_function}'))

    def activate(self, input_values):
        # prepend bias node to layer inputs
        self.input_values = np.vstack([np.array([[1.0]]),
                                       col_vector(input_values)])
        self.signal = col_vector(np.transpose(self.weights)
                                 @ self.input_values)
        self.activation = self.theta(self.signal)
        return self.activation

    def initialize_delta(self, y):
        self.delta = col_vector(2 * (self.activation - y)
                                * self.theta_prime(self.signal))
        return self.delta, self.weights[1:]  # leave out the bias weight

    def compute_delta(self, delta_next, weights_next):
        self.delta = col_vector(self.theta_prime(self.signal)
                                * (weights_next @ delta_next))
        return self.delta, self.weights[1:]  # leave out the bias weight

    def update_weights(self, eta):
        gradient = self.input_values @ np.transpose(self.delta)
        self.weights = self.weights - eta * gradient
        return self.weights


class NeuralNetwork:
    """NeuralNetwork."""

    def __init__(self,
                 shape,
                 weights=None,
                 eta=1,
                 verbose=False):
        self.eta = eta  # learning rate for weight update
        self.verbose = verbose
        # initialize Network Layers
        self.layers = []
        if type(shape) not in [list, tuple]:
            raise ValueError('invalid format for shape')
        self.num_inputs = shape[0][0]
        for i in range(1, len(shape)):
            # preset imported weights, if any
            set_weights = None
            if weights is not None:
                set_weights = weights[i-1]
            self.layers.append(
                NeuralNetworkLayer(
                    inputs=shape[i-1][0],
                    nodes=shape[i][0],
                    activation_function=shape[i][1],
                    layer_num=i,
                    weights=set_weights,
                    output_layer=True if i == len(shape)-1 else False,
                    verbose=self.verbose,
                )
            )

    def fit(self,
            x_train,
            y_train,
            epochs=1,
            eta=None):
        indexes = np.arange(len(x_train))
        for epoch in range(0, epochs):
            # run one iteration over the entire training set in random order
            np.random.shuffle(indexes)
            for i in indexes:
                x_this = col_vector(x_train[i])
                y_this = col_vector(y_train[i])
                # forward-propagate activation
                self.activate_network(x_this)
                # backpropagate error
                delta = None
                weights = None
                for layer in reversed(self.layers):
                    if layer.output_layer:
                        delta, weights = layer.initialize_delta(y_this)
                    else:
                        delta, weights = layer.compute_delta(delta, weights)
                # update weights according to error
                my_eta = eta or self.eta  # optional learning rate override
                for layer in self.layers:
                    layer.update_weights(my_eta)

            if self.verbose:
                # print out error for prediction with current weights
                y_pred = self.predict(x_train)
                mse = mean_squared_error(y_train, y_pred)
                print(f'EPOCH {epoch + 1}, error: {mse}')

    def predict(self, x):
        res = None
        for xi in x:
            activation = self.activate_network(xi)
            if res is None:
                res = activation
            else:
                res = np.vstack((res, activation))
            pass
        return res

    def activate_network(self, x):
        if len(x) != self.num_inputs:
            raise ValueError((f'invalid length of input vector '
                              f'({len(x)}, expected {self.num_inputs})'))
        activation = x.copy()  # first 'activation' is the inputs
        for layer in self.layers:
            activation = layer.activate(input_values=activation)
        # reshape result because we're stacking row vectors for the result set
        return activation.reshape(1, activation.shape[0])


def get_activation_function(act):
    """Provide function and derivative of activation function."""
    if act not in [
        'tanh',
        'sigmoid',
        'relu',
        'lrelu',
    ]:
        raise ValueError(f'Invalid activation function: {act}')

    if act == 'tanh':
        return (
            lambda _: np.tanh(_),
            lambda _: 1 - (np.tanh(_)) ** 2,
        )
    elif act == 'sigmoid':
        def sigm(x):
            return 1 / (1 + np.exp(-x))
        return (
            sigm,
            lambda _: sigm(_) * (1 - sigm(_)),
        )
    elif act == 'relu':
        return (
            lambda _: np.maximum(0, _),
            np.vectorize(lambda _: 0.0 if _ < 0.0 else 1.0),
        )
    elif act == 'lrelu':
        return (
            lambda _: np.maximum(0.01 * _, _),
            np.vectorize(lambda _: 0.01 if _ < 0.0 else 1.0),
        )
    return None


def col_vector(x):
    """Turn scalars and single-dimension arrays into column vector arrays."""
    if isinstance(x, (np.number, numbers.Complex)):
        return np.array([[x]])
    elif isinstance(x, np.ndarray) and len(x.shape) == 1:
        return x.reshape(x.shape[0], 1)
    return x
