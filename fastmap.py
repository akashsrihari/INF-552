"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#Calculate distance taken from FastMap algorithm
def calculateDistance(i, j, coordinates, k, distance_matrix):
    temp = distance_matrix[i,j]*distance_matrix[i,j]
    from math import sqrt
    while k > 0:
        temp = temp - (coordinates[i, k-1]-coordinates[j, k-1])*(coordinates[i, k-1]-coordinates[j, k-1])
        k = k-1
    temp = sqrt(temp)
    return temp
    
#Find pivot points from distance matrix using heuristic
def findPivotPoints(number_of_objects, distance_matrix, coordinates, k):
    Ob = 0
    Oa = 1
    oldOa = 0
    oldOb = 0
    while oldOa != Oa or oldOb != Ob:
        oldOa = Oa
        oldOb = Ob         
        Oa = 0
        for item in range(number_of_objects):
            if calculateDistance(Ob, item, coordinates, k, distance_matrix)> calculateDistance(Ob, Oa, coordinates, k, distance_matrix):
                Oa = item
        Ob = 0
        for item in range(number_of_objects):
            if calculateDistance(Oa, item, coordinates, k, distance_matrix)> calculateDistance(Oa, Ob, coordinates, k, distance_matrix):
                Ob = item    
        
    return sorted([Oa, Ob])    

#Calculate the coordinates of point i given pivot points a and b 
def calculateCoordinates(distance_matrix, a, b, i, k, coordinates):
    x = calculateDistance(a,i,coordinates,k,distance_matrix)* calculateDistance(a,i,coordinates,k,distance_matrix)
    x += calculateDistance(a,b,coordinates,k,distance_matrix)*calculateDistance(a,b,coordinates,k,distance_matrix)  
    x -= calculateDistance(b,i,coordinates,k,distance_matrix)*calculateDistance(b,i,coordinates,k,distance_matrix)
    x = x/ 2 / calculateDistance(a,b,coordinates,k,distance_matrix)
    return x

#Load data into file  
dimensions = 2
number_of_objects = 10
file_open = open('fastmap-data.txt','r')
data = file_open.read()
data = data.split('\n')
data_list = []
for row in range(0,len(data)):
    if data[row] == '':
        break       
    data_list.append(list(map(int, data[row].split())))
distance_matrix = np.zeros([number_of_objects, number_of_objects])


for row in data_list:
    distance_matrix[row[0]-1][row[1]-1] = row[2]
    distance_matrix[row[1]-1][row[0]-1] = row[2]
    
coordinates = np.empty([number_of_objects, dimensions])
pivot_points = np.empty([dimensions, 2])
col = -1

#FastMap algorithm used to reduce dimensions
def fastMap(col, distance_matrix, number_of_objects):
    global coordinates
    global pivot_points
    if col >= dimensions - 1:
        return
    col += 1
    Oa, Ob = findPivotPoints(number_of_objects, distance_matrix, coordinates, col)
    pivot_points[col,0] = Oa
    pivot_points[col,1] = Ob
    for i in range(number_of_objects):
        coordinates[i, col] = calculateCoordinates(distance_matrix, Oa, Ob, i, col, coordinates)

          
    fastMap(col, distance_matrix, number_of_objects)

#Run FastMap
fastMap(col, distance_matrix, number_of_objects)
print "\nPivot Points:"
print pivot_points
print "\nCoordinates of points:"
print coordinates

#Map coordinates with words
file_open = open('fastmap-wordlist.txt','r')
data = file_open.read()
word_list = data.split()

#Plot words with their coordinates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
x = pd.DataFrame(coordinates[:,0])
x['1'] = pd.DataFrame(coordinates[:,1])
plt.plot(coordinates[:,0], coordinates[:,1],'ro')
for label, x, y in zip(word_list, coordinates[:, 0], coordinates[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()

#PCA data for Fast Map
distance_matrix = np.load("PCA_Matrix.npy")
coordinates = np.zeros((distance_matrix.shape[0],dimensions))
pivot_points = np.zeros((dimensions,2))
number_of_objects = 6000
col=-1
fastMap(col,distance_matrix,number_of_objects)

#Plot output
plt.figure()
print "Plotting Data"
plt.plot(coordinates[:,0],coordinates[:,1],"ro")
plt.show()