#!/usr/bin/env python3
"""
Neural Network with Backpropagation.

Author: Matthias Priesters
"""

import numpy as np
import activation_function as af


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
        self.input_values = input_values
        self.signal = np.transpose(self.weights) @ self.input_values
        if type(self.signal) == np.float64:
            self.signal = np.array([self.signal])
        self.activation = self.theta.function(self.signal)
        if type(self.activation) == np.float64:
            self.activation = np.array([self.activation])
        print('weights: ' + str(self.weights))
        print('output: ' + str(self.activation))
        return self.activation

    def initialize_delta(self, y):
        self.delta = (2 * (self.activation - y)
                      @ self.theta.derivative(self.signal))
        if type(self.delta) == np.float64:
            self.delta = np.array([self.delta])
        print('delta: ' + str(self.delta))
        return self.delta, self.weights[1:]

    def compute_delta(self, delta_next, weights_next):
        self.delta = (self.theta.derivative(self.signal)
                      * (weights_next @ delta_next))
        if type(self.delta) == np.float64:
            self.delta = np.array([self.delta])
        print('delta: ' + str(self.delta))
        return self.delta, self.weights[1:]

    def update_weights(self):
        return self.weights


class NeuralNetwork:
    """NeuralNetwork."""

    def __init__(self,
                 shape,
                 weights=None,
                 activation_function='tanh'):
        # choose proper Activation Function
        if activation_function == 'tanh':
            self.activation_function = af.TanHyp()
        elif activation_function == 'sigmoid':
            self.activation_function = af.Sigmoid()
        elif activation_function == 'relu':
            self.activation_function = af.Relu()
        else:
            raise ValueError((f'Unknown Activation Function '
                              f'"{activation_function}"'))
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
        # TODO: for each iteration, pick random X (-> SGD)
        # TODO: for now only one item:
        x = x_train[0]
        y = y_train[0]

        # calculate current prediction
        self.activate_network(x)

        # backpropagate error
        # iterating backwards through the layers
        delta = None
        weights = None
        for layer in reversed(self.layers):
            if layer.output_layer:
                delta, weights = layer.initialize_delta(y)
            else:
                delta, weights = layer.compute_delta(delta, weights)
        return delta

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
            # prepend bias node to layer inputs
            input_values = np.concatenate((np.array([1.0]), input_values))
            input_values = layer.activate(input_values=input_values)
        return input_values


if __name__ == '__main__':
    assert False, 'Not intended for standalone execution.'
