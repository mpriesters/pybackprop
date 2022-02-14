"""
Neural Network with Backpropagation.

References
----------
Abu-Mostafa, Y. S., Magdon-Ismail, M., & Lin, H.-T. (2012).
*Learning from data: A short course.* AML.
"""

# Author: Matthias Priesters

from sklearn.metrics import mean_squared_error
import numpy as np
import numbers


class NeuralNetworkLayer:
    """Neural Network Layer.

    Parameters
    ----------
    inputs : int
        Number of input values to this layer.
    nodes : int
        Number of nodes in this layer.
    activation_function : {'tanh', 'sigmoid', 'relu', 'lrelu'}
        Activation function to use in this layer.
    layer_num : int
        Number of the layer in the network.
    weights : np.ndarray
        Optional weights matrix to initialize this layer with.
        Weights are initialized randomly if empty.
    output_layer : bool
        Marks this layer as final layer of the network whose
        activation is the network's output.
    verbose : bool
        Enables verbose text output.

    Attributes
    ----------
    theta : function
        Activation function.
    theta_prime : function
        First derivative of the activation function.
    input_values : np.ndarray
        Inputs to this layer, including bias term.
    signal : np.ndarray
        Inputs to the activation function given the input values and weights.
    activation : np.ndarray
        Outputs of the activation function given the signal.
    delta : np.ndarray
        Sensitivity vector for error propagation.
    """

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
        """Calculate the activation of the layer by its input values.

        Parameters
        ----------
        input_values : np.ndarray

        Returns
        -------
        np.ndarray
        """
        # prepend bias node to layer inputs
        self.input_values = np.vstack([np.array([[1.0]]),
                                       col_vector(input_values)])
        self.signal = col_vector(np.transpose(self.weights)
                                 @ self.input_values)
        self.activation = self.theta(self.signal)
        return self.activation

    def initialize_delta(self, y):
        """Initialize the sensitivity vector for the output layer.

        Parameters
        ----------
        y : np.ndarray
            Target output.

        Returns
        -------
        tuple[np.ndarray]
            Computed sensitivity vector and current weights matrix
            of this layer.
        """
        self.delta = col_vector(2 * (self.activation - y)
                                * self.theta_prime(self.signal))
        return self.delta, self.weights[1:]  # leave out the bias weight

    def compute_delta(self, delta_next, weights_next):
        """Compute the sensitivity vector for a hidden layer.

        Parameters
        ----------
        delta_next : np.ndarray
            Sensitivity vector of following layer.
        weights_next : np.ndarray
            Weights matrix of following layer.

        Returns
        -------
        tuple[np.ndarray]
            Computed sensitivity vector and current weights matrix
            of this layer.
        """
        self.delta = col_vector(self.theta_prime(self.signal)
                                * (weights_next @ delta_next))
        return self.delta, self.weights[1:]  # leave out the bias weight

    def update_weights(self, eta):
        """Update weights matrix for this layer.

        Parameters
        ----------
        eta : float
            Learning rate.

        Returns
        -------
        np.ndarray
            Updated weights matrix of this layer.
        """
        gradient = self.input_values @ np.transpose(self.delta)
        self.weights = self.weights - eta * gradient
        return self.weights


class NeuralNetwork:
    """Neural Network.

    Parameters
    ----------
    shape : list[list[int or str]]
        Defines the shape of the network. Each element must contain the
        number of nodes as the first element, each element except for the
        first must contain a string defining an activation function (valid
        values: 'tanh', 'sigmoid', 'relu', 'lrelu').
        Example: [[5], [10, 'tanh'], [3, 'tanh']].
    weights : list[np.ndarray]
        Optional preset of weights. Must contain an element for each
        hidden layer.
    eta: float
        Learning rate for weight update.
    verbose: bool
        Enables verbose text output.

    Attributes
    ----------
    layers : list[NeuralNetworkLayer]
        Layers of this network.
    num_inputs : int
        Number of input nodes (i.e. length of first layer).
    """

    def __init__(self,
                 shape,
                 weights=None,
                 eta=1,
                 verbose=False):
        self.eta = eta
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
        """Fit the network to training data by learning appropriate weights.

        Parameters
        ----------
        x_train : np.ndarray
            Input values of the training set.
        y_train : np.ndarray
            Target output values of the training set.
        epochs : int
            Number of iterations to run on the training set.
        eta : float
            Learning rate.
        """
        indexes = np.arange(len(x_train))
        for epoch in range(0, epochs):
            # run one iteration over the entire training set in random order
            np.random.shuffle(indexes)
            for i in indexes:
                x_this = col_vector(x_train[i])
                y_this = col_vector(y_train[i])
                # forward-propagate activation
                self._activate_network(x_this)
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
        """Calculate predictions for a set of input items.

        Parameters
        ----------
        x : np.ndarray
            Set of data points to be predicted.

        Returns
        -------
        np.ndarray
            Prediction for input data.
        """
        res = None
        for x_i in x:
            activation = self._activate_network(x_i)
            if res is None:
                res = activation
            else:
                res = np.vstack((res, activation))
        return res

    def _activate_network(self, x):
        """Forward-propagate activation for one input item.

        Parameters
        ----------
        x : np.ndarray
            Data point to be predicted.

        Returns
        -------
        np.ndarray
            Prediction for input data.
        """
        if len(x) != self.num_inputs:
            raise ValueError((f'invalid length of input vector '
                              f'({len(x)}, expected {self.num_inputs})'))
        activation = x.copy()  # first 'activation' is the inputs
        for layer in self.layers:
            activation = layer.activate(input_values=activation)
        # reshape result because we're stacking row vectors for the result set
        return activation.reshape(1, activation.shape[0])


def get_activation_function(act):
    """Provide function and derivative of activation function.

    Parameters
    ----------
    act : {'tanh', 'sigmoid', 'relu', 'lrelu'}
        Name of activation function. Can be one of 'tanh', 'sigmoid',
        'relu', 'lrelu'.

    Returns
    -------
    tuple[function]
        Tuple of functions containing the function itself and
        its first derivative.
    """
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
    """Turn scalars and single-dimension arrays into column vector arrays.

    Parameters
    ----------
    x : np.number or numbers.Complex or np.ndarray

    Returns
    -------
    np.ndarray
        Input reshaped as column vector.
    """
    if isinstance(x, (np.number, numbers.Complex)):
        return np.array([[x]])
    elif isinstance(x, np.ndarray) and len(x.shape) == 1:
        return x.reshape(x.shape[0], 1)
    return x
