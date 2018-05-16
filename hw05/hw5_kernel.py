# Task c and d

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
import random
# Requires running of the first programming cell beforehand!!!
# Firstly for imports and secondly for the linear k-means algorithm

# k function (Gaussian RBF kernel)
def k_func(x, xp, sigma):
    return np.exp(- np.linalg.norm(x - xp) ** 2 / (2 * sigma ** 2))


def kmeans_kernel(given_k, data, sigma):
    # Gaussian RBF kernel
    
    # Initialize random classes
    z = np.zeros((len(data), given_k))
    init_random_z = np.random.randint(given_k, size=len(data))
    for i in range(0, len(z)):
        z[i, init_random_z[i]] = 1
    
    iterations = 0
    not_converged = True
    while(not_converged):
        iterations+=1
        # Loop to K
        z_new = np.zeros((len(data), given_k))
        distances = np.zeros((len(data), given_k))
        for k in range(0, given_k):
            N_k = np.sum(z[:,k])
            print(N_k)
            
            # Create the ml sum
            ml_sum = 0
            for m in range(0, len(data)):
                for l in range(0,len(data)):
                    ml_sum += z[m,k] * z[l,k] * k_func(data[m], data[l], sigma)
                    
            # N_k ^ -2  times the sum
            third_val = ml_sum * (1 / (N_k * N_k))
            
            
            # Loop through the points
            for n in range(0, len(data)):
                
                # Create the m sum
                m_sum = 0
                for m in range(0, len(data)):
                    m_sum += z[m,k] * k_func(data[n],data[m], sigma)
                
                # N_k ^ -1 times the sum
                second_val = m_sum *(2 / (N_k))
                
                # Calculate the final distances for each point
                distances[n,k]= k_func(data[n],data[n], sigma) - second_val + third_val
        
        tmp = 100 # Arbitrary big number
        # Loop through the points to find which class (k) has the shortest distance
        for index in range(0, len(data)):
            for k in range(0, given_k):
                if tmp > distances[index,k] :
                    tmp = distances[index,k] # Pick that class and keep checking
                    sol_k = k            
            z_new[index,sol_k] = 1     # Change z for the final class to 1
        
        # Check if the new z is equal to the old one
        not_converged = not (np.array_equal(z, z_new))
        z = z_new
        
    plt.figure(figsize=(10,10))
    
    # Plot each of the clusters 
    legend = []
    for i in range(0,given_k):
        indices = [a for a, x in enumerate(z[:,i]) if x == 1]
        plt.scatter(data[indices, 0], data[indices, 1])
        leg = ("Cluster %i" % (i+1))
        legend.append(leg)
        
    # Making the plot pretty
    print("Finished the Gaussian RBF-kernel algorithm in %i iterations." % iterations)
    plt.title("Gaussian RBF kernel K-means cluster algorithm with k = %i and sigma = %.1f." % (given_k, sigma))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(legend)
    plt.show()
    
                
# Loading the data
mat = scipy.io.loadmat('hw5_p1b.mat')
X = mat['X']
# Running the linear kernel from above (Omitting the circles for difference between iterations in task b)

#kmeans(2, X, difference_two_iterations=False)

# Running the gaussian RBF kernel
kmeans_kernel(2,X , 0.2)


