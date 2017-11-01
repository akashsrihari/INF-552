"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np
import random

#Load data from text file
fd = open("linear-regression.txt","r")
text = fd.read()
text = text.split("\n")
ip = []
for i in range(len(text)):
	if text[i]== '' or text[i] == ' ':
		continue
	ip.append(map(float,text[i].split(",")))

#Initialize inputs, labels, weights and intercepts
ip = np.array(ip).T
y = ip[2,:]
ip = np.delete(ip,2,0)
m = ip.shape[1]
d = ip.shape[0]
w = np.random.rand(d,1)
b = random.random()
a=0.1
iter = 7000
cost = 10

#Update weights and intercepts using Learning algorithm
for i in range(iter):
	if cost == 0:
		break
	op = (np.dot(np.transpose(w),ip)+b)
	dz = op - y
	cost = (1.0/m)*np.sum(dz)
	dw = (1.0/m) * np.dot(ip,dz.T)
	db = (1.0/m) * np.sum(dz)
	w = w - (a*dw)
	b = b - (a*db)

print "Weights obtained:\n",w.reshape(2,)
print "Intercept obtained:\n",b

cost =0 
op = (np.dot(np.transpose(w),ip)+b)[0]
for i in range(m):
	cost += op[i]-y[i]
print "Cost - ", cost

#SciKit Learn Implementation
from sklearn.linear_model import LinearRegression 
lr = LinearRegression()
lr.fit(ip.T,y)
print "Weights by sklearn:\n",lr.coef_
print "Intercept by sklearn:\n",lr.intercept_
print "Difference in weights of implementation and sklearn:\n",w - lr.coef_.reshape(d,1)
print "Difference in Intercepts of implementation and sklearn:\n",b - lr.intercept_

w=lr.coef_.reshape(d,1)
b=lr.intercept_
cost = 0
op = (np.dot(np.transpose(w),ip)+b)[0]
for i in range(m):
	cost += op[i]-y[i]
print "SciKit Cost - ", cost
