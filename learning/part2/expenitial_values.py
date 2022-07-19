import math
import numpy as np
#regular normal expression only works with single layers
def exp(inputs):
    exp_values = []
    for output in inputs:
        exp_values.append(E**output)

    print(exp_values)

    norm_base = sum(exp_values)
    norm_values = []
    for value in exp_values:
        norm_values.append(value / norm_base)

    print(norm_values)
    print(sum(norm_values))

#regular normal expression with numpy
def num_exp(input):
    # Can work with batches
    exp_values = np.exp(input)

    print(np.sum(input, axis=1, keepdims=True))

    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)


    print(norm_values)
    # print(sum(norm_values))