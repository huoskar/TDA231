# Task c and d

# k function (RBF kernel)
def k_func(x, xp, sigma):
    return np.exp(- np.linalg.norm(x - xp) ** 2 / (2 * sigma ** 2))


def kmeans_kernel(given_k, data, sigma):
    # Gaussian RBF kernel
    
    #initialize random classes
    z = np.zeros((len(data), given_k))
    init_random_z = np.random.randint(given_k, size=len(data))
    for i in range(0, len(z)):
        z[i, init_random_z[i]] = 1
    not_converged = True
    while(not_converged):
        # Loop to K
        z_new = np.zeros((len(data), given_k))
        distances = np.zeros((len(data), given_k))
        for k in range(0, given_k):
            N_k = np.sum(z[:,k])
            
            # create the ml sum
            ml_sum = 0
            for m in range(0, len(data)):
                for l in range(0,len(data)):
                    ml_sum += z[m,k] * z[l,k] * k_func(data[m], data[l], sigma)
            third_val = ml_sum * (1 / (N_k * N_k))
            # Loop to N
            for n in range(0, len(data)):
                m_sum = 0
                for m in range(0, len(data)):
                    m_sum += z[m,k] * k_func(data[n],data[m], sigma)
                second_val = m_sum *(2 / (N_k))
                
                distances[n,k]= k_func(data[n],data[n], sigma) - second_val + third_val
        
        tmp = 100 # Arbitrary big number
        for index in range(0, len(data)):
            for k in range(0, given_k):
                if tmp > distances[index,k] :
                    tmp = distances[index,k]
                    sol_k = k
            z_new[index,sol_k] = 1
            
        not_converged = not (np.array_equal(z, z_new))
        z = z_new
        
    plt.figure(figsize=(10,10))
    legend = []
    for i in range(0,given_k):
        indices = [a for a, x in enumerate(z[:,i]) if x == 1]
        plt.scatter(data[indices, 0], data[indices, 1])
        leg = ("Cluster %i" % (i+1))
        legend.append(leg)
    plt.show()
    
                
# Loading the data
mat = scipy.io.loadmat('hw5_p1b.mat')
X = mat['X']
# Running the linear kernel from above (Omitting the circles for difference between iterations in task b)
#kmeans(2, X, difference_two_iterations=False)

# Running the gaussian kernel
kmeans_kernel(2,X , 0.2)
