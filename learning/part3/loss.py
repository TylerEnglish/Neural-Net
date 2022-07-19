'''
With loss we want to know how confident our
network is. Usually we use regression. Regression is
how we outpur a specific value.

the log of something is going to be natural log of e


Loss functions (determine how wrong the model is):

    Categorical Cross Entropy:


                           ^
        L   = - E   Yi,j log(Yi,j) 
         i       j


        simplifieds to this cause of OHE:
                  ^  
        Li = -log(yi,k)

    Mean absolute error:



'''


import numpy as np
from dataset import create_data 
import math

E = math.e

np.random.seed(0)

X, y = create_data(100,3)



b = 5.2
print(np.log(b))
print(E ** 1.6486586255873816)
print('\n\n\n\n')


softmax_output = [.7, .1, .2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)
loss = -math.log(softmax_output[0])
print(loss)

print('\n\n\n\n')

softmax_outputs = np.array([[.7, .1, .2],
                            [.1,.5,.4],
                            [.02,.9,.08]])
class_targets = [0,1,1]

neg_log = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
    ])

average_loss = np.mean(neg_log)
print(average_loss)
