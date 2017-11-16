"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np 
import random

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
	op = 1/(1+np.exp(-(np.dot(np.transpose(w),ip)+b)))
	cost = (1.0/m)*np.sum(-(y*np.log(op) + (1-y)*np.log(1-op)))
	dz = op - y
	dw = (1.0/m) * np.dot(ip,dz.T)
	db = (1.0/m) * np.sum(dz)
	w = w - (a*dw)
	b = b - (a*db)

print "Weights obtained:\n",w
print "Intercept obtained:\n",b

#SciKit Learn Implementation
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=7000)
lr.fit(ip.T,y)
print "Weights by sklearn:\n",lr.coef_
print "Intercept by sklearn:\n",lr.intercept_

cor = 0
op = 1/(1+np.exp(-(np.dot(np.transpose(w),ip)+b)))[0]
op[op>=0.5]=1
op[op<0.5]=0
for i in range(m):
	cor += 1 if op[i]==y[i] else 0
print "Correct Classifications - ", cor
print op

w = lr.coef_.reshape(3,1)
b = lr.intercept_
cor = 0
op = 1/(1+np.exp(-(np.dot(np.transpose(w),ip)+b)))[0]
op[op>=0.5]=1.0
op[op<0.5]=0.0
for i in range(m):
	cor += 1 if op[i]==y[i] else 0
print "Correct Classifications by SciKit Learn - ", cor