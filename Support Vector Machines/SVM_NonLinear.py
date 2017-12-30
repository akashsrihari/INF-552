"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.axes import Axes                           
from matplotlib.lines import Line2D

#Read and preprocess data
fd = open("nonlinsep.txt","r")
lis = fd.read().split("\n")
lis1=[]
for i in lis:
	if i=="" or i==" ":
		continue
	lis1.append(i.split(","))

X = np.array(lis1).astype(float)
X1 = X[X[:,2]==1][:,0:2]
X2 = X[X[:,2]==-1][:,0:2]
y = X[:,2].reshape(1,100)
X = np.delete(X,2,axis=1).T
alpha = np.random.uniform(-1000.0,1000.0,(1,100))

#Polynomial Kernel Function
k = np.dot(X.T,X)+1
k = np.dot(k.T,k)

#Equation to be maximized
def maximizer(alpha):
	return (np.sum(np.dot(np.dot((alpha*y).T,alpha*y).T,k))/2.0) - np.sum(alpha)

#Map X to new space using basis function
def mapper(X):
	mapp = np.ones((6,X.shape[1]))
	mapp[1,:] = (2**0.5)*(X[0,:])
	mapp[2,:] = (2**0.5)*(X[1,:])
	mapp[3,:] = (2**0.5)*(X[1,:])*(X[0,:])
	mapp[4,:] = (2**0.5)*(X[0,:])*(X[0,:])
	mapp[5,:] = (2**0.5)*(X[1,:])*(X[1,:])
	return mapp

#Maximize alpha values
cons = ({'type':'eq','fun': lambda alpha: np.sum(alpha*y)})
bnds = ((0,1),)*100
res = minimize(maximizer,alpha,bounds=bnds,constraints=cons)
alpha = np.array(res.x).reshape(1,100)
i1 = np.argmax(alpha[0])
xt_basis = mapper(X[:,i1].reshape(2,1))
w = np.sum(alpha*y*mapper(X),axis=1)
w0 = y[0,i1] - np.dot(w.T,xt_basis)

new_op = np.dot(w.T,mapper(X))+w0
new_op[new_op>=0]=1
new_op[new_op<0]=-1
print "Weights are:\n",w
print "Intercept is:\n",w0
print "Accuracy on training data: ",(0.5*np.sum(abs(new_op-y)))/100.0

from sklearn.svm import SVC
svc = SVC(kernel="poly",degree=2)
svc.fit(X.T,y.reshape(100,))
print "SciKit-Learn's Coefficient:\n",svc.dual_coef_
print "SciKit-Learn's Intercept:\n",svc.intercept_
print "SciKit-Learn's Accuracy: ",svc.score(X.T,y.reshape(100,))

"""
#Data Visualization
plt.figure()
plt.plot(X1[:,0],X1[:,1],'ro')
plt.plot(X2[:,0],X2[:,1],'bo')
plt.show()
"""