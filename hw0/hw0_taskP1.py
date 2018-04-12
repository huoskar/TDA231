import numpy as np
import matplotlib.pyplot as plt

# Given
mean = [1, 1]
cov = [[0.1, -0.05], [-0.05, 0.2]]
p = np.random.multivariate_normal(mean, cov, 1000)


def func(x, r):
    temp = np.subtract(x, mean)
    upper = np.dot(np.dot(np.matrix.transpose(temp), np.linalg.inv(cov)), temp)
    return upper / 2 - r


# Gather points
min = -0.5
max = 2.5
res = 200
x = np.linspace(min, max, res)
y = np.linspace(min, max, res)
X, Y = np.meshgrid(x, y)

# Calculate different levels, and which points should be blue/black
levels = []
blue = []
black = []

for r in range(1, 4):
    level = []
    for i in range(len(x)):
        rows = []
        for j in range(len(y)):
            point = [x[i], y[j]]
            rows.append(func(point, r))
        level.append(rows)
    levels.append(level)

    for k in range(len(p)):
        result = func(p[k], r)
        if result > 0 and r == 3:
            black.append(p[k])
        else:
            blue.append(p[k])

# Use the layers to plot the level sets
plt.contour(X, Y, np.array(levels[0]), [0], colors='yellow')
plt.contour(X, Y, np.array(levels[1]), [0], colors='orange')
plt.contour(X, Y, np.array(levels[2]), [0], colors='red')

# Scatter plot the blue and black points respectively on top of the level sets
plt.scatter([row[1] for row in blue], [row[0] for row in blue], color='blue')
plt.scatter([row[1] for row in black], [row[0] for row in black], color='black')
plt.title("%i points lie outside  $f(x,3)=0$ depicted as black dots" % len(black))
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()