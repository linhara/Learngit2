
def createLayers(neuronMatr, input):
    for n, ip in neuronMatr[0], input:
        n.inputneuron(ip)
    for layer in neuronMatr[1:]:
        for n, ip in layer, input:
            n.initNeuron(ip)


import numpy as np
inp = np.array([1,2,3, 3, 4])
rows = 4
col = 5

def main():
    weights = np.random.randn(rows, col)
    #print(weights)
    #print("svar1: ")                                    #vadsom x N dot N x vadsom, .T är oviktigt
    #print(np.dot(weights, inp))
    #activations = inp
    #biases = np.random.randn(rows)
    #weights = np.random.randn(rows, col)
    #activations = stepForward(activations, weights, biases)

    for i in range(150):
        print(np.random.randn())

    for i in range(150):
        print(np.random.randn())

def xd():
    x=1


#main()