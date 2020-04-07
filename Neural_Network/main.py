import numpy as np
import Layers
inp = np.array([1, 2, 3, 3, 4]).T
structList = [len(inp), 6, 4, 2]
expected = [0.5, 1]
rows = 4
col = 5
layers = []
costList = []
activationOfEachLayer = [inp]

def main():
    initStruct()
    runNetwork(inp)               #Ger en lista med activations från varje layer, activationOfEachLayer
    costList.append(calcCost(activationOfEachLayer[-1], expected)) #ingen aning om de begöver sammlas i lista
    #-------------------------


    print(activationOfEachLayer)




    # print(layer.biases)
    # print(layers[0].weights)

def initStruct():
    for i in range(len(structList[:-1])):
        layers.append(Layers.layer(structList[i+1], structList[i]))


def runNetwork(inp):
    activations = inp
    for layer in layers:
        activations = layer.stepForward(activations)[0]
        activationOfEachLayer.append(activations)


def calcCost(ans, expected):
    return sum((ans - expected)**2)


def costPrime(ans, expected):
    return 2*(ans - expected)


if __name__ == '__main__':
    main()