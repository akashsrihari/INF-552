"""

Created by:

Srihari Akash Chinam (2953497706 - chinam@usc.edu)

and 

Sayan Nanda (2681592859 - snanda@usc.edu)

"""

from __future__ import division
import numpy as np


file_open = open("hmm-data.txt", 'r')
data = file_open.read()
gwlines = data.split('\n')[2:12]
#GRID_WORLD
grid_world = []
for line in gwlines:
    grid_world.append(map(int,line.split()))

#STATE_SPACE
state_space = []
for i in range(10):
    for j in range(10):
        state_space.append([i,j])


def init_prob(gridw):
    summation = 0
    final_mat = [[0]*10]*10
    for row in gridw:
        summation += sum(row)
    for row in range(len(gridw)):
        for elem in range(len(gridw[row])):
            final_mat[row][elem] = float(gridw[row][elem])/summation
    return final_mat


#INITIAL PROBABILITIES
initial_probabilities = init_prob(grid_world)

tower_locations = [[0,0],[0,9],[9,0],[9,9]]
nd = data.split('\n')[24:35]

#OBSERVATIONS
observations = []
for line in nd:
    observations.append(map(float,line.split()))


def trans_prob(gridw_list):
    gridw = np.array(gridw_list)
    #print gridw
    final_mat = np.zeros(shape = (len(gridw)*len(gridw[0]),len(gridw)*len(gridw[0])))
    padded_mat = np.zeros(shape = (len(gridw)+2,len(gridw[0])+2))
    padded_mat[1:len(gridw)+1,1:len(gridw)+1] = gridw
    wid = len(gridw)
    for i in range(1,len(gridw)+1):
        for j in range(1,len(gridw)+1):
            count = 0
            if padded_mat[i,j-1] == 1:
                count = count + 1
            if padded_mat[i-1,j] == 1:
                count = count + 1
            if padded_mat[i,j+1] == 1:
                count = count + 1
            if padded_mat[i+1,j] == 1:
                count = count + 1
            if count == 0:
                chance = 0
            else:
                chance = float(1.0/count)

            if padded_mat[i,j-1] == 1:
                final_mat[(i - 1) * wid + (j - 1), (i - 1) * wid + (j - 2)] = chance
            if padded_mat[i-1,j] == 1:
                final_mat[(i - 1) * wid + (j - 1), (i - 2) * wid + (j - 1)] = chance
            if padded_mat[i,j+1] == 1:
                final_mat[(i - 1) * wid + (j - 1), (i - 1) * wid + (j)] = chance
            if padded_mat[i+1,j] == 1:
                final_mat[(i - 1) * wid + (j - 1), (i) * wid + (j - 1)] = chance

    return final_mat.tolist()


#TRANSITION MATRIX
transition_matrix = trans_prob(grid_world)


def distance_measure(X,Y):
    from math import sqrt
    return sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)


def poss_range(gridw, tower_loc, statspace):
    from math import floor
    ret = [[] for _ in xrange(len(statspace))]
    for i in range(len(statspace)):
        temp2 = [[] for _ in xrange(len(tower_loc))]
        temp = []
        for j in range(len(tower_loc)):
            dist = distance_measure(tower_loc[j],state_space[i])
            temp.append(floor(dist * 0.7 *10)/10)
            temp.append(floor(dist * 1.3 *10)/10)
            temp2[j] = temp
            temp = []
        ret[i] = temp2
    return ret


#POSSIBLE RANGES
possible_range = poss_range(grid_world,tower_locations, state_space)

#B_MATRIX
def B_matrix(ind, obs, statspace, tower_loc, pos_range):
    for j in range(len(tower_loc)):
        if (pos_range[ind][j][0] <= obs[j] and obs[j] <= pos_range[ind][j][1]) == False:
            return 0

    count = 0
    for i in range(len(statspace)):
        flag = False
        for j in range(len(tower_loc)):
            if pos_range[i][j][0] <= obs[j] and obs[j] <= pos_range[i][j][1]:
                flag = True
            else:
                flag = False
        if flag == True:
            count = count + 1
    return (1/float(count))

#print B_matrix(44,observations[0], state_space, tower_locations, possible_range)


def maximum1(T1, i, j, k,trans_mat):
    maxim = 0
    for c1 in range(k):
        temp = T1[c1][i-1] * trans_mat[c1][j] * B_matrix(j, observations[i], state_space, tower_locations, possible_range)
        if temp > maxim:
            maxim = temp

    return maxim


def argmax1(T1, i, j, k, trans_mat):
    maxim = 0
    maxim_val = -1
    for c1 in range(k):
        temp = T1[c1][i - 1] * trans_mat[c1][j] * B_matrix(j, observations[i], state_space, tower_locations,
                                                           possible_range)
        if temp > maxim_val:
            maxim_val = temp
            maxim = c1
    return maxim

def argmax2(T1,t, k):
    amax = 0
    vmax = -1
    for i in range(k):
        temp = T1[i][t-1]
        if temp > vmax:
            vmax = temp
            amax = i

    return amax


#VITERBI ALGORITHM
k = len(state_space)
t = len(observations)
Z = [[] for _ in xrange(t)]
X = [[] for _ in xrange(t)]
T1 = [[[]for _ in range(t)] for _ in xrange(k)]
T2 = [[[]for _ in range(t)] for _ in xrange(k)]
for i in range(k):
    T1[i][0] = initial_probabilities[int(i/10)][i%10] * B_matrix(i,observations[0],state_space,tower_locations,possible_range)
    T2[i][0] = 0


for i in range(1,t):
    for j in range(k):
        T1[j][i] = maximum1(T1, i, j, k, transition_matrix)
        T2[j][i] = argmax1(T1, i, j, k, transition_matrix)


Z[t-1] = argmax2(T1, t, k)
X[t-1] = Z[t-1]

for i in range(t-1,0, -1):
    Z[i-1] = T2[Z[i]][i]
    X[i-1] = Z[i-1]

final_X = []
for i in range(len(X)):
    final_X.append((int(X[i]/10),X[i]%10))
print final_X