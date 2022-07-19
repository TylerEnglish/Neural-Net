"""
Activation Function:
    Comes into play after inputs*weights + bias
    This is whats feed into the activation function
    Every neuron after input layer is probably going to have an activation function
    Without our activation would just be a linear unit
    With linear activation we can only fit linear function
    We need non-linear activation to be able to fit non-linear functions better

Step function:
    y = { 1    x > 0 
        { 0    x <= 0


Rectified Linear Unit (reLU) function:
    y = {x     x > 0
        {0     x <= 0

    1. Fast
    2. Works


Sigmoid Function:
             1
   y =  -----------
          1 + e^-x

    vanishing gradiant problem

Exponential Function:
    y = e^x

Softmax Function: (we want to determine how correct our output is)

                    e^z 
                        i,j
    S       =  _________________
      i,j         L
                E       e^Z
                  l=1      i,j


"""
import numpy as np
from dataset import create_data 
import math

E = math.e

np.random.seed(0)

X, y = create_data(100,3)

"""
#reLu example
for i in inputs:
    output.append(max(0, i))

print(output)
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

class ReLU_Act:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

dense1 = Layer_Dense(2, 3)
activation1 = ReLU_Act()

dense2 = Layer_Dense(3,3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

