import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/greedo/Documents/TDA231/dataset0.txt")
# print (data.shape)
# print (data[0])

newdata = np.zeros(data.shape)
for col in range(0, data.shape[1]):
    max = np.max(data[:, col])
    newdata[:, col] = data[:, col] / max

Y = newdata.T
X = data.T
cov = np.cov(X)
cov_scaled = np.cov(Y)

# Find features with minimim correlation
min_value = cov_scaled.min()
print("The minimum correlation has a value of: %f" % min_value)

index, _ = np.where(cov_scaled == min_value)
print("The features with least correlation are feature: %i and %i" % (index[0], index[1]))

feature_col1 = Y[index[0], :]
feature_col2 = Y[index[1], :]

# Scatter plot of the two features
plt.figure(1)
plt.scatter(feature_col1, feature_col2)
plt.title("The features with least correlation: %i and %i, with a value of %f" % (index[0], index[1], min_value))
plt.xlabel("Feature %i" % index[0])
plt.ylabel("Feature %i" % index[1])
plt.show()

# Covariance of data
plt.figure(2)
plt.imshow(cov, interpolation='nearest')
plt.colorbar()
plt.title("Covariance of data")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
# Covariance of normalized data
plt.figure(3)
plt.imshow(cov_scaled, interpolation='nearest')
plt.colorbar()
plt.title("Covariance of normalized data")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

# Dessa 2 ser konstiga ut
# Correlation of data
plt.figure(4)
plt.imshow(np.corrcoef(X))
plt.colorbar()
plt.title("Correlation of data")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
# Correlation of normalized data
plt.figure(5)
plt.imshow(np.corrcoef(Y))
plt.colorbar()
plt.title("Correlation of normalized data")
plt.xlabel("First feature")
plt.ylabel("Second feature")

plt.show()