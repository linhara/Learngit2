import numpy as np
class layer:
    def __init__(self, nrOfNeurons, weightsPerNeuron):
        self.biases = np.random.randn(nrOfNeurons)
        self.weights = np.random.randn(nrOfNeurons, weightsPerNeuron)

    def stepForward(self, activations):
        return self.sigmoid(np.dot(self.weights, activations) + self.biases.T)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
