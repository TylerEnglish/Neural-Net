"""
Difference here from the neuron worksheet is 
that we would be is we are calculating a full 
layer
"""

#Example of how a layer would work
inputs = [6.8, 1.6, 2.1]

"""
We need 3 seprate weights list to account for the 
3 nerouns we are inputing into. As before in the neuron 
example, each weight and bias is unique to it
"""
weights1 = [7.9, 1.1, 7.4]
weights2 = [-1.92, 4.9, .92]
weights3 = [0.99, -5.4, 2.1]

bias1 = 3
bias2 = 2
bias3 = .02

"""
The output of our layer should look similar to 
what the inputs looks like which is why I put it
as a list. 
"""
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + bias1, 
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + bias3]

print("Output:\n",output, "\n\n\n")


# Simplified version same of whats above

weights = [[7.9, 1.1, 7.4],
           [-1.92, 4.9, .92],
           [0.99, -5.4, 2.1]]

biases = [3,2,.02]




layer_output = [] # where we want our output to be stored
for n_weight, n_bias in zip(weights, biases):
    n_output = 0 # Output of given neuron
    for n_inputs, weight in zip(inputs, n_weight):
        n_output += n_inputs * weight
    n_output += n_bias
    layer_output.append(n_output)

print("Simplified Output:\n",layer_output)