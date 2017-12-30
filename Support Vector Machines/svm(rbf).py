"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

from __future__ import division
import numpy as np
from scipy import optimize
import math
import matplotlib.pyplot as plt

#Read and Preprocess Data
file_open = open("nonlinsep.txt", 'r')
data = file_open.read()
lis = []
for line in data.split():
    lis.append(map(float,line.split(',')))
X = np.array(lis)
Y = X[:, 2]
X = X[:, :2]
Y = Y.reshape(1,100)
alpha = np.zeros(len(X)).reshape(100,1)
def dist(X1,X2):
    return (X2[1] - X1[1])**2 + (X2[0]-X1[0])**2


#Kernel Function for RBF
k = np.zeros(len(X)*len(X)).reshape(len(X),len(X))
for i in range(len(X)):
    for j in range(len(X)):
        k[i,j] = math.exp((-1)*(dist(X[i],X[j])))
Q = k
Q = Q*Y*Y.T

#Function to be maximized to find alpha
def L(alpha):
    return 0.5*np.dot(np.dot(np.transpose(alpha),Q), alpha) - np.sum(alpha)

#Bounds and constraints for alpha
lam = 0.001
bnds = ((0, (1/100/2/lam)),) * 100
cons = ({'type': 'eq', 'fun':lambda x: np.dot(Y,x)})

res_cons = optimize.minimize(L, alpha, bounds = bnds, constraints = cons)
alpha = res_cons.x
alpha = alpha.reshape(100,1)
for i in range(0,len(alpha)):
    if alpha[i] < 0.0001:
        alpha[i] = 0

#Kernel Function for calculating RBF after alpha is known
kx = np.zeros(len(X)).reshape(len(X),1)
for i in range(len(X)):
    kx[i,0] = math.exp((-1)*(dist(X[i],[0,0])))


#Weight for the curve
W = np.sum(alpha.T * Y * kx.T, axis = 1)
print "Weight-", W

#Constant to be added to the curve
imax = np.argmax(alpha)
B = -W*kx[imax,0]
print "b-", B

#Check for Accuracy against Training Data
count = 0
for i in range(len(X)):
    yexpected = Y[0,i]
    yobserved = W*kx[i,0] + B
    if yobserved <= 0:
        yobserved = -1
    else:
        yobserved = 1
    if yobserved == yexpected:
        count += 1
print count