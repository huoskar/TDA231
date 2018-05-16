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

#Empty list for containing the class points.
classlist = [[]] * k
for i in range(0, len(X)):
    class_ = find_closest_class(mu, X[i])
    classlist[class_].append(X[i].tolist())
#print(classlist[0])

for class_ in classlist:
    print[class_[0][:]]
    plt.scatter(class_[:][0], class_[:][1])
#plt.scatter(X[:, 0], X[:, 1])
#plt.scatter(mu[:, 0], mu[:, 1])
plt.show()
