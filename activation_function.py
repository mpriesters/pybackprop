#!/usr/bin/env python3
"""
Activation Functions for a Neural Network with Backpropagation.

Author: Matthias Priesters (mpriesters@users.noreply.github.com)
"""

from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """Abstract Activation Function class."""

    @abstractmethod
    def function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class TanHyp(ActivationFunction):
    """Hyperbolic Tangent Activation Function."""

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - (self.function(x))**2


class Sigmoid(ActivationFunction):
    """Logistic Sigmoid Activation Function."""

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        # source: https://towardsdatascience.com/
        #         derivative-of-the-sigmoid-function-536880cf918e
        return self.function(x) * (1 - self.function(x))


class Relu(ActivationFunction):
    """Rectified Linear Unit Activation Function."""

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        if x < 0:
            return 0
        if x > 0:
            return 1
        # not differentiable at zero
        return np.nan


if __name__ == '__main__':
    assert False, 'Not intended for standalone execution.'
