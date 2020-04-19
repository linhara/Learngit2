import numpy as np
a = np.array([0.5, 1, 10, 100, 1000])
b = np.array([1, 2, 3, 4])
m = np.array([[2, 2, 2, 20],
              [2, 2, 2, 2],
              [2, 2, 2, 2]])
listy = [1, 2, 3, 4]
c = np.array([[1], [2], [3], [4], [5], [6]])            #är 6 x 1 matris
d = np.array([[10, 20, 30]])                            #är 1 x 3 materis, för att de kan matrix multipliceras
list1 = [1,2,3]
list2 = [3,4,5]

def main():
    #print(sigmoid(a))           #numpy smart
    #print(a**2)
    #print(sum(a))
    #print(a * b)
    #print(np.ones((a.shape[0], 1)))
    #print(np.hstack(([1], a)))
    #print(1 - a)
    #print(np.dot(m,b))
    print(np.dot(d,1))



def sigmoid(x):
    return 1/(1 + np.exp(-x))

main()