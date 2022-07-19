"""
So instead of only using a single 
layer of inputs we will be using 
multiple layers of inputs which is 
defined as a batch

Batches does:
1. Since we are doing it in parallel the more parallel operations we can run
2. Reason why we would want to run it on GPU instead of GPU
3. Helps with Generalizations
"""
import numpy as np
"""
I'm changing the shapes of input and 
weights to a 3,4 so we can learn how to
work with different methods with numpy
"""
inputs = [[6.8, 1.6, 2.1, 6.2],
          [0.3, 2.1, -.92, .6],
          [-1.1, 2.5, 6.01, -.01]]

"""
We don't need to add more weights 
or biases because we aren't adding
more neurons. All we adding is more
inputs which is batches
"""
weights = [[7.9, 1.1, 7.4, 1.1],
           [-1.92, 4.9, .92, -2.8],
           [0.99, -5.4, 2.1, 6.7]]

biases = [3,2,.02]

"""
The .dot is now using the Matric Product method.

What we needed to do was transpose the weight array. 
"""
output = np.dot(inputs, np.array(weights).T) + biases
print("Batches neuron output using dotproduct:\n",output, "\n\n")


