"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np 
import random
import matplotlib.pyplot as plt
import sys
import copy
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

#Initialize inputs, labels, weights and intercepts
ip = np.delete(np.array(ip),3,1)
y = ip[:,3]
y[y==-1]=0
ip = np.transpose(np.delete(ip,3,1))
m = ip.shape[1]
ip = np.append(ip,np.ones(shape=(1,m))).reshape(4,m)
d = ip.shape[0]
w = np.zeros(shape=(d,))
a=0.01
it = 7000
mi = sys.maxint
logger = []

"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
w1 = pca.fit_transform(ip[0:3,:].T)
w1 = np.append(w1,y.reshape(2000,1),axis=1).reshape(2000,3)
x1 = w1[w1[:,2]==1]
x2 = w1[w1[:,2]==0]
x1 = x1[:,0:2]
x2 = x2[:,0:2]


plt.figure()
plt.plot(x1[:,0],x1[:,1],"yo")
plt.plot(x2[:,0],x2[:,1],"ro")
plt.show()
"""

start=time.time()

#Update weights and intercepts using Learning algorithm
for j in range(it):
	sum_err=0
	op = np.dot(w.reshape(1,4),ip)[0]
	op[op>=0]=1
	op[op<0]=0
	err = (y - op)
	for i in range(m):
		w = w + err[i]*(a*ip[:,i])
	sum_err = np.sum(abs(err))
	if sum_err<mi:
		mi = sum_err
		wbest = copy.deepcopy(w)
	logger.append(sum_err)
end = time.time()

print("Time taken: %.2f seconds"%(end-start))
print "Best case scenario weights:\n",wbest[0:-1]
print "Best case scenario intercept:\n",wbest[-1]

cor = 0
op = np.dot(w.reshape(1,4),ip)[0]
op[op>=0]=1
op[op<0]=0
for i in range(m):
	if op[i]==y[i]:
		cor+=1
print "Correct Classifications - ", cor

#Plot data for misclassifications per iteration
for i in range(14):
	plt.figure()
	plt.plot(range(500*i,500*(i+1)),logger[500*i:500*(i+1)])
	plt.show()