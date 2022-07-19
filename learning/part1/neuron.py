"""
Every Neuron has a unique connection to every
single previous neuron. 
Every previous neuron outputs becomes this neuron
input. 
Every input has a unique weight
Every unique neuron has a unique biase
"""


#Example of how a neuron would work without activation function

inputs = [6.8, 1.6, 2.1]
weights = [7.9, 1.1, 7.4]

#only 1 bias because we only looking at 1 neuron which means 1 bias
bias = 3

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)