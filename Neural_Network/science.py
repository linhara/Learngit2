import numpy as np
# define the sigmoid function
def sigmoid(x, derivative=False):

    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

# choose a random seed for reproducible results
np.random.seed(1)

# learning rate
alpha = .1

# number of nodes in the hidden layer
num_hidden = 3

# inputs
inputs = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
])

# outputs
# x.T is the transpose of x, making this a column vector
expected = np.array([[0, 1, 0, 1, 1, 0]]).T

# initialize weights randomly with mean 0 and range [-1, 1]
# the +1 in the 1st dimension of the weight matrices is for the bias weight
hidden_weights = 2 * np.random.random((inputs.shape[1] + 1, num_hidden)) - 1
output_weights = 2 * np.random.random((num_hidden + 1, expected.shape[1])) - 1

# number of iterations of gradient descent
num_iterations = 1

# for each iteration of gradient descent
for i in range(num_iterations):

    # forward phase
    # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
    input_layer_outputs = np.hstack((np.ones((inputs.shape[0], 1)), inputs))
    #print(input_layer_outputs)
    hidden_layer_outputs = np.hstack((np.ones((inputs.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, hidden_weights))))
    print(hidden_layer_outputs)
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)

    # ---------------------backward phase--------------------------------------
    # output layer error term
    output_error = 2*(output_layer_outputs - expected)
    # hidden layer error term
    # [:, 1:] removes the bias term from the backpropagation        big parenthesis is derivitive of sigmoid(x)
    hidden_error = (hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:])) * np.dot(output_error, output_weights.T[:, 1:])
    #(sigprim)*np.dot(WHICHERRORHERE??, next_layer_weights)
    #print(output_weights.T[:, 1:])                                                              #6 x 1            1 x 3
    #print(hidden_error)
    #print((hidden_layer_outputs[:, 1:]))

    #print((1 - hidden_layer_outputs[:, 1:]))

    # partial derivatives
    output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]     #np.newaxis är eftersom deras input är en matris
    hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]

    # average for total gradients
    total_output_gradient = np.average(output_pd, axis=0)
    total_hidden_gradient = np.average(hidden_pd, axis=0)
    #print(hidden_pd)
    #print(total_hidden_gradient)
    #print(hidden_pd)
    #print(total_hidden_gradient)

    # update weights
    hidden_weights += - alpha * total_hidden_gradient
    output_weights += - alpha * total_output_gradient


# print the final outputs of the neural network on the inputs X
print("Output After Training: \n{}".format(output_layer_outputs))