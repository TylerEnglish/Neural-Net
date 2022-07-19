"""
Working with what numpy can do
Using shape, vectors, tensor

An example of how we want our 
layers to look like
"""
import numpy as np

inputs = [6.8, 1.6, 2.1]
weight = [7.9, 1.1, 7.4]

bias = 3

"""
The .dot is multipling and adding
each value

So basically:
6.8*7.9 + 1.6*1.1 + 2.1*7.4

Then we add the bias
"""
output = np.dot(weight, inputs) + bias
print("Single neuron output using dotproduct:\n",output, "\n\n")


weights = [[7.9, 1.1, 7.4],
           [-1.92, 4.9, .92],
           [0.99, -5.4, 2.1]]

biases = [3,2,.02]
"""
What's happening here is:
np.dot(weights, inputs) = [np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)]

Then we add both vectors to get output like this:
np.dot(weights, inputs) + biases
"""
output = np.dot(weights, inputs) + biases
print("Layered neuron output using dotproduct:\n",output, "\n\n")