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
fd = open("linsep.txt","r")
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
alpha = np.random.randn(1,100)

#Function to be maximized for alpha
def maximizer(alpha):
	return -1*(((-0.5)*np.sum(np.dot((alpha*y*X).T,alpha*y*X)))+np.sum(alpha))

#Set constraints and bounds on alpha
cons = ({'type':'eq','fun': lambda alpha: np.sum(y*alpha)})
bnds = ((0,None),)*100
res = minimize(maximizer,alpha,bounds=bnds,constraints=cons)
alpha = np.array(res.x)
w = np.sum(alpha*y*X,axis=1)
i1=np.argmax(alpha)
b = y[0,i1] - np.dot(w,X[:,i1])
wmod=np.dot(w.T,w)**0.5
print "Weights obtained:\n",w
print "Intercept obtained:\n",b
op = y*(np.dot(w.T,X)+b)/wmod
margin = min(op[0])

#Create SVM model for given data
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X.T,y.reshape(100,))
print "SciKit-Learn coefficients:\n",svc.coef_
print "SciKit-Learn Intercept:\n",svc.intercept_
w1 = svc.coef_
b1 = svc.intercept_

#Function to calculate points in the line
def pointer(w0,w1,x):
	return (w1*x)+w0

b = b/w[1]
m = -w[0]/w[1]
b1 = b1/w1[0][1]
m1 = -w1[0][0]/w1[0][1]

#Plot data
fig = plt.figure()
ax = Axes(fig, [.1,.1,.8,.8])
fig.add_axes(ax)
l = Line2D([0, 1],[pointer(b,m,0),pointer(b,m,1)],color='black')
ax.add_line(l)
l = Line2D([0, 1],[pointer(b1,m1,0),pointer(b1,m1,1)],color='green')
ax.add_line(l)
plt.plot(X1[:,0],X1[:,1],'bo')
plt.plot(X2[:,0],X2[:,1],'ro')
plt.show()