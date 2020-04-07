import numpy as np
a = np.array([0.5, 1, 10, 100])
b = np.array([2, 2, 2, 2])

def main():
    #print(sigmoid(a))           #numpy smart
    #print(a**2)
    #print(sum(a))
    print(a * b)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

main()