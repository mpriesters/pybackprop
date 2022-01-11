#!/usr/bin/env python3
"""
Neural Network with Backpropagation.

Author: Matthias Priesters
"""

from activation_function import vectorize, TanHyp, Sigmoid, Relu
import numpy as np


class NeuralNetworkLayer:
    """NeuralNetworkLayer."""

    def __init__(self,
                 inputs,
                 nodes,
                 activation_function,
                 layer_num,
                 weights=None,
                 output_layer=False):
        # initialize weights randomly between -0.01 and 0.01 if no weights are given
        if weights is not None:
            self.weights = weights
        else:
            self.weights = 0.02 * np.random.rand(inputs + 1, nodes) - 0.01
        # static settings for this layer
        self.theta = activation_function
        self.layer_num = layer_num
        self.output_layer = output_layer
        # work variables for this layer
        self.input_values = None
        self.signal = None
        self.activation = None
        self.delta = None
        print((f'initializing layer {layer_num} with {inputs} inputs, '
               f'{nodes} nodes, function {self.theta.__class__.__name__}'))

    def activate(self, input_values):
        # prepend bias node to layer inputs
        self.input_values = np.vstack([np.array([[1.0]]),
                                       vectorize(input_values)])
        self.signal = vectorize(np.transpose(self.weights)
                                @ self.input_values)
        self.activation = self.theta.function(self.signal)
        #print('weights: ' + str(self.weights))
        #print('output: ' + str(self.activation))
        return self.activation

    def initialize_delta(self, y):
        self.delta = vectorize(2 * (self.activation - y)
                               @ self.theta.derivative(self.signal))
        #print('delta: ' + str(self.delta))
        return self.delta, self.weights[1:]  # leave out the bias weight

    def compute_delta(self, delta_next, weights_next):
        self.delta = vectorize(self.theta.derivative(self.signal)
                               * (weights_next @ delta_next))
        #print('delta: ' + str(self.delta))
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
                 activation_function='tanh',
                 eta=1,
                 k=1):
        # choose proper Activation Function
        if activation_function == 'tanh':
            self.activation_function = TanHyp()
        elif activation_function == 'sigmoid':
            self.activation_function = Sigmoid()
        elif activation_function == 'relu':
            self.activation_function = Relu()
        else:
            raise ValueError((f'Unknown Activation Function '
                              f'"{activation_function}"'))
        self.eta = eta  # learning rate for weight update
        self.k = k  # maximum number of training iterations
        # initialize Network Layers
        self.layers = []
        if type(shape) not in [list, tuple]:
            raise ValueError('invalid format for shape')
        self.num_inputs = shape[0]
        for i in range(1, len(shape)):
            # preset imported weights, if any
            set_weights = None
            if weights is not None:
                set_weights = weights[i-1]
            self.layers.append(
                NeuralNetworkLayer(
                    inputs=shape[i-1],
                    nodes=shape[i],
                    activation_function=self.activation_function,
                    layer_num=i,
                    weights=set_weights,
                    output_layer=True if i == len(shape)-1 else False,
                )
            )

    def fit(self, x_train, y_train):
        # TODO: do this in a loop until the error becomes small
        for k in range(0, self.k):
            # fit to random data point
            i = np.random.randint(len(x_train))
            x = x_train[i]
            y = y_train[i]

            print(f'==========\nEPOCH {k}')
            print(f'i: {i}, x: {x}, y: {y}')
            # calculate current prediction
            prediction = self.activate_network(x)

            # backpropagate error
            # iterating backwards through the layers
            delta = None
            weights = None
            for layer in reversed(self.layers):
                if layer.output_layer:
                    delta, weights = layer.initialize_delta(y)
                else:
                    delta, weights = layer.compute_delta(delta, weights)
            # update weights according to error
            for layer in self.layers:
                upd_weights = layer.update_weights(self.eta)
                #print(upd_weights)
            print(f'---> prediction: {prediction}')

    def predict(self, x):
        res = np.array([])
        for xi in x:
            activation = self.activate_network(xi)
            res = np.append(res, activation)
        return res

    def activate_network(self, x):
        if len(x) != self.num_inputs:
            raise ValueError((f'invalid length of input vector '
                              f'({len(x)}, expected {self.num_inputs})'))
        input_values = x.copy()
        for layer in self.layers:
            input_values = layer.activate(input_values=input_values)
        return input_values


if __name__ == '__main__':
    assert False, 'Not intended for standalone execution.'
