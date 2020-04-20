import numpy as np
import Layers
inp = np.array([1, 2, 3, 3, 4]).T
structList = [len(inp), 5, 3]
expected = [0, 1, 0]
layers = []
activation_of_each_layer = [inp]
learn_rate = 0.1
#inputs = [[0, 0, 1],[0, 1, 1],[1, 0, 0],[1, 1, 0],[1, 0, 1],[1, 1, 1]]

def main():
    init_struct()
    run_network(inp)               #Ger en lista med activations från varje layer, activationOfEachLayer
    cost = calc_cost_prime(activation_of_each_layer[-1], expected)
    #print(layers)
    # -------------------------back_prop------------

    cor_output_layer(cost)
    for index_of_layer in reversed(range(len(layers)-1)):
        back_prop(index_of_layer)


    # ----------------------------------------------

    print(activation_of_each_layer)
    print(cost)


def init_struct():
    for i in range(len(structList[:-1])):
        layers.append(Layers.layer(structList[i], structList[i+1]))


def run_network(inp):
    activations = inp
    for layer in layers:
        activations = layer.step_forward(activations)
        activation_of_each_layer.append(activations)        #do i ever use this shit???


def back_prop(i):
    layers[i].error = layers[i].sig_prime() * np.dot(layers[i+1].error, layers[i+1].weights[:, 1:])
    hopefully_gradient = layers[i].received_activations[:1] * layers[i].error
    layers[i].weights += -learn_rate * hopefully_gradient[:, np.newaxis]

def cor_output_layer(cost):                                         #denna funktionen borde inte behövas
    layers[-1].error = cost
    hopefully_gradient = layers[-1].received_activations[:1] * cost
    layers[-1].weights += -learn_rate * hopefully_gradient[:, np.newaxis]

def calc_cost_prime(ans, expected):
    return 2*(ans - expected)


if __name__ == '__main__':
    main()