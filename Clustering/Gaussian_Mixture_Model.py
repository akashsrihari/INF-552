"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import math
import random
import copy
import numpy as np
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture


#Calculate Gaussian of a point given mean and covariance matrices
def gaus(x,mean,covmat,dim):
    xt = np.matrix(x-mean).transpose()
    covinv = np.linalg.inv(covmat)
    x = np.matrix(x-mean)
    return pow(pow(2*math.pi,dim)*np.linalg.det(covmat),-0.5)*math.exp((-0.5)*(x*covinv*xt))

#Calculate weights(Probability) given amplitude and gaussian values
def calc_Prob(amp,gaus,numClusts):
    probs = []
    tot = 0.0
    for i in range(numClusts):
        tot+=amp[i]*gaus[i]
    for i in range(numClusts):
        probs.append(float(amp[i]*gaus[i])/tot)
    return probs

#Calculate euclidean distance between two points
def dist(p1,p2):
    return pow(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2),0.5)

#Segregate all points into clusters using k-means
def kmeans(centroids, cluster_list, numClust):
    old_centroids = copy.deepcopy(centroids)-1     
    while abs((old_centroids-centroids).sum()) > 0: 
        old_centroids = copy.deepcopy(centroids)
        groups=[]
        for i in range(numClust):
            groups.append([])
        for i in range(len(cluster_list)):
            mini = 10000
            ind=0
            for j in range(numClust):
                x = dist(centroids[j],cluster_list[i])
                if x<mini:
                    ind=j
                    mini=x
            groups[ind].append(cluster_list[i])
        for j in range(numClust):
            groups[j] = np.matrix(groups[j])
            lenger = groups[j].shape[0]
            centroids[j] = np.sum(groups[j],axis=0)/lenger
    return groups

#Read data from text file
print "\nPreprocessing..."
numClust = 3
txt_object = open("clusters.txt","r")
txt_data = txt_object.read()
clusters = txt_data.split("\n")
cluster_list = []
for pair in clusters:
    temp = []
    temp.append(float(pair.split(',')[0]))
    temp.append(float(pair.split(',')[1]))
    cluster_list.append(temp)
dataMatrix = np.array(cluster_list)
leng = dataMatrix.shape[0]
dims = dataMatrix.shape[1]
    
#Generate random clusters for k-means
print "\nExecuting kmeans to initialise weights..."
centroids = []
from random import randint
ints = []
for i in range(0, numClust):
    x = randint(0,len(cluster_list)-1)
    while x in ints:
        x = randint(0,len(cluster_list)-1)
    centroids.append(cluster_list[x])

#Create cluster groups
centroids = np.array(centroids)
cluster_list = np.array(cluster_list)
groups = kmeans(centroids,cluster_list,numClust)
    
#Initialize weights using clusters generated from kmeans
print "\nInitializing weights..."
weights = np.zeros((leng,numClust))
for i in range(leng):
    for j in range(numClust):
        if dataMatrix[i] in groups[j]:
            weights[i][j]=1

#Initialize means, covariances and amplitudes
means = np.random.random((numClust,dims))
amp = np.random.random((numClust))
covars=[]
for i in range(numClust):
    covars.append(np.random.random((dims,dims)))

#Begin Expectation Maximization process 
print "\nBeginning Expectation Maximization..."
num_it=0
logli=10000
all_logger=[]
while True:    
    num_it += 1
    old_means = copy.deepcopy(means)
    old_amp = copy.deepcopy(amp)
    old_covars=copy.deepcopy(covars)
    sums = np.sum(weights,axis=0)
    #print num_it, sums
    all_sum = sums.sum()
    
    #Expectation Process
    for j in range(numClust):
        amp[j]=sums[j]/all_sum
        summer=np.zeros((dims))
        for k in range(dims):
            for i in range(leng):
                summer+=weights[i][j]*dataMatrix[i]
            means[j] = summer/sums[j]
        d = copy.deepcopy(dataMatrix)
        d = np.subtract(d,means[j])
        dw = copy.deepcopy(d)
        for i in range(leng):
            dw[i] *= weights[i][j]
        covars[j] = (((np.matrix(d).transpose())*np.matrix(dw))/sums[j])
    
    #Calculate Log-likelihood
    old_log = logli
    logli=0.0
    for i in range(leng):
        si=0.0
        for j in range(numClust):
            si += weights[i][j]*amp[j]
        logli += math.log(si)
    #print -logli
    logger=[]
    logger.append(-logli)
    all_logger.append(logger)
    if abs(old_log-logli)<1.0e-7:
        print "Done"
        break
    
    #Maximization Process
    for i in range(leng):
        gausser=np.zeros((3))
        for j in range(numClust):
            gausser[j] = gaus(dataMatrix[i],means[j],covars[j],dims)
        probs = np.array(calc_Prob(amp,gausser,numClust))
        weights[i] = probs

#Outputs of means, covariances and amplitudes
for j in range(numClust):
    print "\nMean of cluster ",j,"\n",means[j]
    print "\nCovariance Matrix of cluster ",j,"\n",covars[j]
    print "\nAmplitude of cluster ",j,"\n",amp[j]


#Visualization process
color_iter = itertools.cycle(['navy', 'cornflowerblue', 'gold'])
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


covars[0] = np.linalg.inv(covars[0])
covars[1] = np.linalg.inv(covars[1])
covars[2] = np.linalg.inv(covars[2])
gmm = mixture.GaussianMixture(n_components=3,covariance_type="full",weights_init=amp,means_init=means,precisions_init=covars).fit(dataMatrix)
plot_results(dataMatrix, gmm.predict(dataMatrix), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')
plt.show()

all_logger = np.array(all_logger)
plt.figure()
for i in range(all_logger.shape[0]):
    plt.plot(all_logger[i][0])
plt.plot(all_logger)
plt.show()