"""
Working with multi layers 
"""
import numpy as np


inputs = [[6.8, 1.6, 2.1, 6.2],
          [0.3, 2.1, -.92, .6],
          [-1.1, 2.5, 6.01, -.01]]

#first layer
weights = [[7.9, 1.1, 7.4, 1.1],
           [-1.92, 4.9, .92, -2.8],
           [0.99, -5.4, 2.1, 6.7]]

biases = [3,2,.02]


"""
Layer 1 output is the output of layer1
but becomes the inputs of layer 2
"""
layer1_output = np.dot(inputs, np.array(weights).T) + biases

#second layer
weights2 = [[.01, 2.1, -.91],
           [3.8, -2, .66],
           [4.5, .72, -1.8]]

biases2 = [1, 5, -.02]

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print("Layering:\n",layer1_output)
