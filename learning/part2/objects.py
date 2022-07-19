"""
Putting what's done so far as an object
"""
import numpy as np

np.random.seed(0)

"""
Usually our feature sets are denoted with the
value X

X = input data
"""
X = [[6.8, 1.6, 2.1, 6.2],
     [0.3, 2.1, -.92, .6],
     [-1.1, 2.5, 6.01, -.01]]


"""
Usually with weights we want the
values to be random between -1 and 1


Generally with Neural Networks we 
want small values

Usually we want to scale our data down
to be in range of -1 and 1
"""
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        n_inputs is size of inputs we want
        n_neurons how many neurons we want
        """
        self.weights = .1 * np.random.randn(n_inputs, n_neurons) #shape is n_inputs by n_neurons
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer_1 = Layer_Dense(4, 5)
layer_2 = Layer_Dense(5, 2)

layer_1.forward(X)
# print(layer_1.output)

layer_2.forward(layer_1.output)
print(layer_2.output)