#!/usr/bin/env python3
"""
Neural Network with Backpropagation.

Author: Matthias Priesters (mpriesters@users.noreply.github.com)
"""

import numpy as np
import activation_function as af


class NeuralNetworkLayer:
    """NeuralNetworkLayer."""

    def __init__(self,
                 inputs,
                 nodes,
                 activation_function,
                 output_node=False):
        # initialize weights randomly between -0.01 and 0.01
        self.weights = 0.02 * np.random.rand(inputs + 1, nodes) - 0.01
        self.theta = activation_function
        self.output_node = output_node
        self.signal = None
        self.activation = None
        print((f'initializing layer with {inputs} inputs, {nodes} nodes, '
               f'function {self.theta.__class__.__name__}'))

    def activate(self, input_values):
        self.signal = np.transpose(self.weights) @ input_values
        self.activation = self.theta.function(self.signal)
        print(self.weights)
        print(self.activation)
        return self.activation


class NeuralNetwork:
    """NeuralNetwork."""

    def __init__(self,
                 shape,
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
            self.layers.append(
                NeuralNetworkLayer(
                    inputs=shape[i-1],
                    nodes=shape[i],
                    activation_function=self.activation_function,
                )
            )
        # add a final one-node layer as output
        self.layers.append(
            NeuralNetworkLayer(
                inputs=shape[len(shape)-1],
                nodes=1,
                activation_function=self.activation_function,
                output_node=True,
            )
        )

    def fit(self, X_train, Y_train):
        pass

    def predict(self, X):
        if len(X) != self.num_inputs:
            raise ValueError((f'invalid length of input vector '
                              f'({len(X)}, expected {self.num_inputs})'))
        for layer in self.layers:
            input_values = np.concatenate((np.array([1.0]), X))
            X = layer.activate(input_values=input_values)
        return X


if __name__ == '__main__':
    assert False, 'Not intended for standalone execution.'
