"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import re

#Read images using this function
def read_file(filename):
	im = cv2.imread(filename,0)
	im = (im - np.mean(im))/np.std(im)
	return im

#Load filenames of training data
fd = open("downgesture_train.list","r")
lis = fd.read()
lis = lis.split("\n")
X = []
y = []
counter = 0
#Load training data
for i in lis:
	if i == "" or i == " ":
		continue
	X.append(read_file(i))
	if "down" in i:
		y.append(1)
	else:
		y.append(0)
	counter+=1

m=184
X_train = np.array(X).reshape(184,960).T
y_train = np.array(y).reshape(1,184)


#Inialize weights of neural network in vectorised manner for faster implementation
w1 = np.random.rand(100,960)
w2 = np.random.rand(1,100)
b1 = np.random.rand(100,1)
b2 = np.random.rand(1,1)

iters = 1000
learning_rate=0.1

#Begin training on network
for i in range(iters):

	#Forward Propagation
	Z1 = np.dot(w1,X_train)+b1
	A1 = 1/(1+np.exp(-Z1))
	Z2 = np.dot(w2,A1) + b2
	A2 = 1/(1+np.exp(-Z2))
	
	#Backward Propagation
	dZ2 = A2-y_train
	err = 0.5*np.sum(dZ2*dZ2)
	dW2 = np.dot(dZ2,A1.T)
	db2 = np.sum(dZ2,axis=1,keepdims=True)
	dZ1 = np.dot(w2.T, dZ2) * A1 * (1-A1)
	dW1 = np.dot(dZ1,X_train.T)
	db1 = np.sum(dZ1,axis=1,keepdims=True)

	#Weights update
	w1 -= learning_rate*dW1
	w2 -= learning_rate*dW2
	b1 -= learning_rate*db1
	b2 -= learning_rate*db2

#Load filenames of testing data
fd = open("downgesture_test.list","r")
lis = fd.read()
lis = lis.split("\n")
X = []
y = []
#Load testing data
for i in lis:
	if i == "" or i == " ":
		continue
	X.append(read_file(i))
	if "down" in i:
		y.append(1)
	else:
		y.append(0)

X_test = np.array(X).reshape(83,960).T
y_test = np.array(y).reshape(1,83)

#Forward propagation of test data to make predictions
Z1 = np.dot(w1,X_test)+b1
A1 = 1/(1+np.exp(-Z1))
Z2 = np.dot(w2,A1) + b2
A2 = 1/(1+np.exp(-Z2))
A2[A2>0.5]=1
A2[A2<=0.5]=0

#Results
incor = np.sum(abs(A2-y_test))
print "Predictions for test files -\n",A2.astype(int)
print "Number of correct classifications - ",(83.0-incor)
print "Accuracy - ",(83.0-incor)/83.0

#SciKit Learns Neural Network
from sklearn.neural_network import MLPClassifier as mlp
m = mlp(activation="logistic", learning_rate_init=0.1,max_iter=1000,tol=0)

#Training of the model
m.fit(X_train.T,y_train.T.reshape(184,))

#Testing of the model
op = m.predict(X_test.T)
op = np.array(op).reshape(1,83)

#Scikit Learn Results
incor = np.sum(abs(op-y_test))
print "Predictions for test files -\n",op
print "Sklearn Number of correct classifications - ",(83.0-incor)
print "Sklearn Accuracy - ",m.score(X_test.T,y_test.reshape(83,))