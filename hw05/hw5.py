import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
import random

mat = scipy.io.loadmat('hw5_p1a.mat')
print (mat.keys())
X = mat['X']

#takes two points and returns the distance netween them as a float
def euclidian_distance(point_origin, point_target):
    delta_distance = [point_origin[0] - point_target[0], point_origin[1] - point_target[1]]
    return math.sqrt(delta_distance[0]**2 + delta_distance[1]**2)

def find_closest_class(mu, point):
    tmp = 0
    for i in range(1, len(mu)):
        if(euclidian_distance(mu[i], point) < euclidian_distance(mu[tmp], point)):
            tmp = i
    return tmp

# k is an arbitrary number of clusters that we wish to classify. Note that 0 is a class, so k = 2 would make 3 classes.

def kmeans(k, data):

    # z[n, k]= 0 means that x[n] does not belong to class k. z[n,k]= 1 means that it does belong to class k
    
    # Guess mu  
    prevmu = np.zeros((k, 2))
    mu = np.zeros((k,2))
    for i in range(0, k):
        mu[i] = [random.uniform(min(data[:, 0]), max(data[:, 0])), random.uniform(min(data[:, 1]), max(data[:, 1]))]
    iterations = 0
    while(not(np.array_equal(prevmu, mu))):
        iterations +=1
        # Loop through points and assign each point to its closest mean
        z = np.zeros((len(data), k))
        for i in range(0, len(data)):   
            class_ = find_closest_class(mu, data[i])
            # Set it in z
            z[i,class_] = 1

        prevmu = np.copy(mu)
        
        for i in range(0,k):
            indices = [a for a, x in enumerate(z[:,i]) if x == 1]
            mu[i] = (np.mean(data[indices,0]),np.mean(data[indices,1]))
 
    
    # Loop through k again to plot the classes
    for i in range(0,k):
        indices = [i for i, x in enumerate(z[:,i]) if x == 1]
        plt.scatter(data[indices, 0], data[indices, 1])

    # Also plot mu
    plt.scatter(mu[:, 0], mu[:, 1] , color='r')
    print(iterations)
    plt.show()

kmeans(3,X)
