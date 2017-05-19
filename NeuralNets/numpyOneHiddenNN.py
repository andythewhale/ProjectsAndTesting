#This is an example of a Neural Net that has 3 inputs 3 out puts and 3 nodes in a hidden layer. Everything is fully connected.

import numpy as np

#These are the input weights. These will feed to the first hidden layer.
#To be 100% clear, thesea are the edges coming from the inputs.
input_weights  = np.array([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]).T

#These are the hidden layer weights, These will feed to the output layer.
#To be 100% clear, these are the edges coming from the hidden nodes.
hidden_weights = np.array([0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]).T

#Storing the weights
weights = [input_weights, hidden_weights]

#Junk arrays to represent input values:
A = np.array([1, 2, 3])
B = np.array([1, 2, 3])
C = np.array([1, 2, 3])

def neural_network(input, weights):

	#multiply our inputs by our input_weights to get our hidden values.
	hidden = input.dot(weights[0])
	#multiply our hidden values by our hidden_weights to get our predicted values.
	pred = hidden.dot(weights[1])

for i in A:

	input = np.array([A[i], B[i], C[i])

	predicted = neural_network(input, weights)
	print(predicted)

#Cool


