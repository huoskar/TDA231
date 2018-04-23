from sklearn import datasets
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import random
from sklearn import model_selection


digits = datasets.load_digits()

data = digits.data
target_names = digits.target_names


#Modified from question 3
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


####a.)
#This function gets the mean of each pixel value in two sets of images and returns it as a 64-float array
def get_mean(fives, eights):
    mu_fives = np.zeros(64)
    for i in range(0, len(fives[0])):
        mu_fives[i] = np.mean(fives[:][i])

    mu_eights = np.zeros(64)
                         
    for i in range(0, len(eights[0])):
        mu_eights[i] = np.mean(eights[:][i])

    return mu_fives, mu_eights


####b.)
#This function takes two sets of images with pixel values between 0-16 and normalizes them to 0-1,
#then finds the variance in each row and column.
def get_normalized_variance(fives, eights):
    fives = np.asarray(fives)
    eights = np.asarray(eights)
    fives = fives/16.0
    eights = eights/16.0
    
    #first 8 values are row variance, last 8 values are column variance
    sigma_fives = np.zeros(16)
    sigma_eights = np.zeros(16)

    #Find the feature vector for all images in fives.
    for i in range(0, len(fives[0])):
        sigma_fives += get_feature_vector(fives[i])

    sigma_fives = sigma_fives / len(fives[0])
    
    for i in range(0, len(eights[0])):
        sigma_eights += get_feature_vector(eights[i])

    sigma_eights = sigma_eights / len(eights[0])
    return sigma_fives, sigma_eights

#Takes an array with 64 values, splits in into a matrix with 8x8 columns and finds
#the variance over each row and each column, which is returned in an array with 16 elements.
#The first 8 elements are the row variances, while the last are the column variances.
def get_feature_vector(data):
    tmp = np.split(data, 8)
    tmp = np.array(tmp)
    x = np.zeros(16)
    x[:8] += tmp.var(axis = 1)
    x[8:] += tmp.var(axis = 0)
    return x

####c.)
fives = data[digits.target == 5]
eights = data[digits.target == 8]

#Create a model to split the data with, shuffle the values so we
#won't get so many consecutive fives and eights.
KF = model_selection.KFold(n_splits=5, shuffle=True)
k = 1

#Merge the fives and eights data
data = np.asarray(fives.tolist() + eights.tolist())

#Split the data into training and testing data based on the model.and start iterating.
for train_index, test_index in KF.split(data):
    training_fives = []
    training_eights = []

    #check every training index.
    for i in train_index:
        #Check if the index in the training data is a five or an eight.
        #The weird boolean is used to compare lists.
        if((fives == data[i]).all(1).any()):
            training_fives.append(data[i])
        else:
            training_eights.append(data[i])
    
    print("==== Iteration %i ====" % k)
    k += 1
    #from a.)
    mu_5, mu_8 = get_mean(training_fives, training_eights)
    #from b.)
    feature_5, feature_8 = get_normalized_variance(training_fives, training_eights)
    
    task_a = 0
    task_b = 0
    n = 0
    for i in test_index:
        n += 1
        a = new_classifier(data[i], mu_5, mu_8)
        #Check f we managed to classify a correct five with the method used in a.)
        if ((fives == data[i]).all(1).any() and a ==1):
            task_a += 1
        #Else, check if it was a correctly classified eight
        elif ((eights == data[i]).all(1).any() and a == -1):
            task_a += 1
        #Get the feature vector for the image
        x = get_feature_vector(data[i])
        b = new_classifier(x, feature_5, feature_8)
        #Check f we managed to classify a correct five with the method used in b.)
        if((fives == data[i]).all(1).any() and b == 1):
            task_b +=1
        #Else, check if it was a correctly classified eight
        elif ((eights == data[i]).all(1).any() and b == -1):
            task_b += 1
    
    print("Accuracy of task a.): %.3f" %(task_a/float(n)) )
    print("Accuracy of task b.): %.3f" %(task_b/float(n)) )


    
    

###
