import numpy as np
from dataset import create_data 
import math

E = math.e

np.random.seed(0)

X, y = create_data(100,3)

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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        #so we don't hit infinite with 0 value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        '''
        Right here we are checking if the 
        values are scaler or not 

        the if shape is 1 checks for scaler
        and tries to make a prediction base off 
        of it

        the else check for non-scalar so like 
        OHE and tries to make a prediction off of
        it
        
        '''
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        
        neg_log_like = -np.log(correct_confidences)

        return neg_log_like



dense1 = Layer_Dense(2, 3)
activation1 = ReLU_Act()

dense2 = Layer_Dense(3,3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_func = Loss_CategoricalCrossentropy()
loss = loss_func.calculate(activation2.output, y)

print(loss)

