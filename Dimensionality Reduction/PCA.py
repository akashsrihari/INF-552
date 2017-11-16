"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import numpy as np
import matplotlib.pyplot as plt

#Number of dimensions
k = 2

#Load data
data = open("pca-data.txt","r")
data = data.read()
data = data.split("\n")
for i in range(len(data)):
	if data[i]=="":
		continue
	data[i] = data[i].strip().split()
for i in range(len(data)):
	for j in range(len(data[i])):
		data[i][j] = float(data[i][j])
data = np.array(data).T

#Calculate covariance matrix and then the Eigen Values
covar = np.cov(data)
eigen_val, eigen_vect = np.linalg.eig(covar)
dicto={}
for i in range(len(eigen_val)):
	dicto[eigen_val[i]] = eigen_vect[:,i]
x = dicto.keys()
x.sort(reverse=True)

#Select the highest eigen values as required dimensions
fin_list = np.zeros((2,3))
print "Eigenvectors for new data"
for i in range(k):
	fin_list[i] = dicto[x[i]]
	print dicto[x[i]]
fin = fin_list.dot(data)

#Visualize data by plotting
plt.figure()
plt.plot(fin.T[:,0],fin.T[:,1],"bo")
plt.show()

#SciKit-Learn implementation of PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
w = pca.fit_transform(data.T)

#Eigenvectors of PCA data by SciKit-Learn
print "Eignevectors for SciKit PCA data:"
print pca.components_

#Visualize output of PCA by SciKit-Learn
plt.figure()
plt.plot(w[:,0],w[:,1],"yo")
plt.show()