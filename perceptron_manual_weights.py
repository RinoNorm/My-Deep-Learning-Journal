import numpy as np
import matplotlib.pyplot as plt

# Classification of points using a perceptron with manually assigned weights
# First, generate random points that will later be classified. No perceptron is involved at this stage
n = 5 # The number of points in ONE class
x1 = np.random.random(size=n) # Generate random x-coordinates
x2 = x1 + [np.random.random() for _ in range(n)] # Generate y-coordinates using the same VALUES (not the same coordinates) as the x-coordinates
# Add another value to each y-coordinate so that the points do not all lie on a straight line
c1 = [x1, x2] # Define the first class

x1 = np.random.random(size=n) # The same process, but for points in the second class
x2 = x1 - [np.random.random() for _ in range(n)] - 0.1 # Subtract values to place the second class below the first
c2 = [x1, x2] # Define the second class

# This is where the perceptron comes into play to classify objects
w = np.array([-0.3, 0.3]) # Assign the weights as usual
for i in range(n): # Iterate over all points in ONE class
    x = np.array([c2[0][i], c2[1][i]]) # Assemble the coordinates of the i-th point in the second class
    y = w @ x
    if y >= 0: # If the weighted sum is nonnegative, assign the point to c1; otherwise, assign it to c2
        print("class c1")
    else:
        print("class c2")
print("\n")

def graph1():
    plt.scatter(c1[0], c1[1], s=10, c="red")
    plt.scatter(c2[0], c2[1], s=10, c="blue")
    plt.plot([0, 1], [0, 1])
    plt.grid()
    plt.show()
graph1()

# If the separating line does not pass through the origin, add a constant term: the boundary is defined by x2 = x1 + b
n = 5
b = 3 # Add an offset
x1 = np.random.random(size=n)
x2 = x1 + [np.random.random() for _ in range(n)] + b # Shift the points of the first class along the y-axis
c1 = [x1, x2]

x1 = np.random.random(size=n)
x2 = x1 - [np.random.random() for _ in range(n)] - 0.1 + b # Shift the points of the second class
c2 = [x1, x2]

w = np.array([-0.5, 0.5, -b*0.5]) # Add another value to the weights to account for the offset
for i in range(n):
    x = np.array([c1[0][i], c1[1][i], 1])
    y = w @ x
    if y >= 0:
        print("class c1")
    else:
        print("class c2")
print("\n")

def graph2():
    plt.scatter(c1[0], c1[1], s=10, c="red")
    plt.scatter(c2[0], c2[1], s=10, c="blue")
    plt.plot([0, 1], [b, 1 + b]) # Change the starting and ending y-coordinates of the separating line
    plt.grid()
    plt.show()
graph2()