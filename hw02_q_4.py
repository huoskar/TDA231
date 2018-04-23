from sklearn import datasets
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import random
from sklearn import model_selection


digits = datasets.load_digits()

data = digits.data
target_names = digits.target_names

def new_classifier(Xtest, mu1, mu2):
    d = len(mu1)
    
    mu = np.zeros(d)
    for i in range(0, d):
        mu[i] = mu1[i] - mu2[i]
        
    b = np.zeros(d)
    
    for i in range(0, d):
        b[i] = 0.5 * (mu1[i] + mu2[i])
    point = Xtest-b
    dist = np.sqrt(np.sum((mu)**2))

    Ytest =np.sign( np.dot(np.transpose(mu),point) / dist)

    return Ytest


def get_mean(fives, eights):

    mu_fives = np.zeros(64)
    for i in range(0, len(fives[0])):
        mu_fives[i] = np.mean(fives[:][i])

    mu_eights = np.zeros(64)
    for i in range(0, len(eights[0])):
        mu_eights[i] = np.mean(eights[:][i])

    return mu_fives, mu_eights


####b.)

def get_normalized_variance(fives, eights):

    fives = np.asarray(fives)
    eights = np.asarray(eights)
    fives = fives/16.0
    eights = eights/16.0
    
#first 8 values are row variance, last 8 values are column variance
    sigma_fives = np.zeros(16)
    sigma_eights = np.zeros(16)

    #Image is split into an 8x8 matrix, then variance is calculated along each axis, summarized and divided by the total amount of images
    #number 8 and 15 will always be 0 (first and last column only contains zeroes)
    for i in range(0, len(fives[0])):
        tmp = np.split(fives[i], 8)
        tmp = np.array(tmp)

        sigma_fives[:8] += tmp.var(axis = 1)
        sigma_fives[8:] += tmp.var(axis = 0)

    sigma_fives = sigma_fives / len(fives[0])

    for i in range(0, len(eights[0])):
        tmp = np.split(eights[i], 8)
        tmp = np.array(tmp)

        sigma_eights[:8] += tmp.var(axis = 1)
        sigma_eights[8:] += tmp.var(axis = 0)

    sigma_eights = sigma_eights / len(eights[0])
    return sigma_fives, sigma_eights

def get_feature_vector(data):
    tmp = np.split(data, 8)
    tmp = np.array(tmp)
    x = np.zeros(16)
    x[:8] += tmp.var(axis = 1)
    x[8:] += tmp.var(axis = 0)
    return x

fives = data[digits.target == 5]
eights = data[digits.target == 8]

print("====== Task3 ======")
KF = model_selection.KFold(n_splits=5, shuffle=True)
k = 1

data = np.asarray(fives.tolist() + eights.tolist())

for train_index, test_index in KF.split(data):
    training_fives = []
    training_eights = []
    
    for i in train_index:
        if((fives == data[i]).all(1).any()):
            training_fives.append(data[i])
        else:
            training_eights.append(data[i])
    
    print("==== Iteration %i ====" % k)
    k += 1
    mu_5, mu_8 = get_mean(training_fives, training_eights)
    feature_5, feature_8 = get_normalized_variance(training_fives, training_eights)
    
    task_a = 0
    task_b = 0
    n = 0
    for i in test_index:
        n += 1
        a = new_classifier(data[i], mu_5, mu_8)
        if ((fives == data[i]).all(1).any() and a ==1):
            task_a += 1
        elif ((eights == data[i]).all(1).any() and a == -1):
            task_a += 1
        x = get_feature_vector(data[i])
        b = new_classifier(x, feature_5, feature_8)
        if((fives == data[i]).all(1).any() and b == 1):
            task_b +=1
        elif ((eights == data[i]).all(1).any() and b == -1):
            task_b += 1
    
    print("Accuracy of task a.): %.3f" %(task_a/float(n)) )
    print("Accuracy of task b.): %.3f" %(task_b/float(n)) )


    
    

###
