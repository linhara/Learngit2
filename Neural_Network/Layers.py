import numpy as np
class layer:
    def __init__(self, nrOfNeurons, weightsPerNeuron):
        self.biases = np.random.randn(nrOfNeurons)
        self.weights = np.random.randn(nrOfNeurons, weightsPerNeuron)

    def stepForward(self, activations):
        Z = np.dot(self.weights, activations) + self.biases.T
        newActivations = self.sigmoid(Z)
        return newActivations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigPrime(self, x):
        return self.sigmoid(x)*(1 - self.sigmoid(x))