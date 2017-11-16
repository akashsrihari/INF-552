"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np 
import random
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import time

#Load data from text file
fd = open("classification.txt","r")
text = fd.read()
text = text.split("\n")
ip = []
for i in range(len(text)):
	if text[i]== '' or text[i] == ' ':
		continue
	ip.append(map(float,text[i].split(",")))

ip = np.delete(np.array(ip),4,1)

"""
#View Data in 3D

x1 = ip[ip[:,3]==1]
x2 = ip[ip[:,3]==-1]
fig = pylab.figure()
ax = Axes3D(fig)
ax.scatter(x1[:,0],x1[:,1],x1[:,2],'y')
ax.scatter(x2[:,0],x2[:,1],x2[:,2],'r')
plt.show()
"""

#Preprocess data
y = ip[:,3]
y[y==-1]=0
ip = np.transpose(np.delete(ip,3,1))
m = ip.shape[1]
ip = np.append(ip,np.ones(shape=(1,m))).reshape(4,m)
d = ip.shape[0]
w = np.zeros(shape=(d,))
a=0.01
checker = True
counter = 0

#View data in 2D
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
w1 = pca.fit_transform(ip[0:3,:].T)
w1 = np.append(w,y.reshape(2000,1),axis=1).reshape(2000,3)
x1 = w1[w1[:,2]==1]
x2 = w1[w1[:,2]==0]
x1 = x1[:,0:2]
x2 = x2[:,0:2]

plt.figure()
plt.plot(x1[:,0],x1[:,1],"yo")
plt.plot(x2[:,0],x2[:,1],"ro")
plt.show()
"""

#Run Perceptron Learning Algorithm
start = time.time()
while checker:
	counter+=1
	sum_err=0
	op = np.dot(w.reshape(1,4),ip)[0]
	op[op>=0]=1
	op[op<0]=0
	err = (y-op)
	for i in range(m):
		w = w + err[i]*(a*ip[:,i])
	sum_err = np.sum(abs(err))
	if sum_err==6:
		checker = False
end = time.time()

print "Time taken: %.2f"%(end-start)
print "Weights obtained:\n",w[0:-1]
print "Intercept: ",w[-1]

#Check for number of correctly classified points
cor = 0
op = np.dot(w.reshape(1,4),ip)[0]
op[op>=0]=1
op[op<0]=0
for i in range(m):
	cor += 1 if op[i]==y[i] else 0
print "Correct Classifications - ", cor

#SciKit Learn Implementation
from sklearn.linear_model import Perceptron
pt = Perceptron()
X = np.transpose(ip)
X = np.delete(X,3,1)
pt.fit(X, y)
print 'Weights, ', pt.coef_
print 'Intercept, ', pt.intercept_
w = np.array(pt.coef_[0])
w = np.append(w, pt.intercept_)
cor = 0
op = np.dot(np.transpose(w.reshape(4,1)),ip)[0]
op[op>=0]=1
op[op<0]=0
for i in range(m):
	if op[i]==y[i]:
		cor+=1
print "Correct Classifications - ", cor