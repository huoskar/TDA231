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
k = 3
# z[n, k]= 0 means that x[n] does not belong to class k. z[n,k]= 1 means that it does belong to class k
z = np.zeros((len(X), k))
#mu are the guessed points
mu = np.zeros((k, 2))
for i in range(0, k):
    mu[i] = [random.uniform(min(X[:, 0]), max(X[:, 0])), random.uniform(min(X[:, 1]), max(X[:, 1]))]

# Loop through points
for i in range(0, len(X)):
    # Find class for point    
    class_ = find_closest_class(mu, X[i])
    # Set it in z
    z[i,class_] = 1
    
# Loop through k again to plot the classes
for i in range(0,k):
    indices = [i for i, x in enumerate(z[:,i]) if x == 1]
    plt.scatter(X[indices, 0], X[indices, 1])

# Also plot mu
plt.scatter(mu[:, 0], mu[:, 1] , color='r')
plt.show()
