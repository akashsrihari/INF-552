"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

import copy
import numpy as np
import sys
import matplotlib.pyplot as plt

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
            groups[j] = np.array(groups[j])
            lenger = groups[j].shape[0]
            centroids[j] = np.sum(groups[j],axis=0)/lenger
    print "Centroids:"
    print centroids
    return groups
  
#Read data from text file    
txt_object = open("clusters.txt","r")
txt_data = txt_object.read()
clusters = txt_data.split("\n")
cluster_list = []
for pair in clusters:
    temp = []
    temp.append(float(pair.split(',')[0]))
    temp.append(float(pair.split(',')[1]))
    cluster_list.append(temp)
    
number_of_clusters = 3
centroids = []

#Select random points as beginner centroids
from random import randint
ints = []
for i in range(0, number_of_clusters):
    x = randint(0,len(cluster_list)-1)
    while x in ints:
        x = randint(0,len(cluster_list)-1)
    centroids.append(cluster_list[x])

centroids = np.array(centroids)
cluster_list = np.array(cluster_list)
groups = kmeans(centroids,cluster_list,number_of_clusters)

#Plot output of kmeans
plt.figure()
color_arr = ['bo','ro','yo']
for j in range(number_of_clusters):
    for k in range(groups[j].shape[0]):
        plt.plot(groups[j][k][0],groups[j][k][1],color_arr[j])
plt.show()