#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


# tells matplotlib to embed plots within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Linear Regression with one variable
# 
# Put data set in a folder named "Data"

# In[8]:


data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X = data[:, 0] 
# First column - input variable

y = data[:, 1]
# Second column - output variable


# In[12]:


m = y.size  # number of training examples
m


# # plot Data

# In[13]:


def plotData(x, y):
   
    '''x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    '''
    
   
    fig = pyplot.figure()  # open a new figure
    
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in $10,000')
    pyplot.xlabel('Population of City in 10,000s')


# In[14]:


plotData(X, y)


# # Cost Function - Add intercept term. 

# In[17]:


X = np.stack([np.ones(m), X], axis=1)


# In[18]:


def computeCost(X, y, theta):
    
    # initialize some useful values
    m = y.size  # number of training examples
    
    # You need to return the following variables correctly
    J = 0
    
    #Hypothesis Function
    h = np.dot(X, theta)
    
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    
    return J


# We still donot know the best value of theta. 
# we randomly give theta values to compute the cost function
# 

# In[19]:


J = computeCost(X, y, theta=np.array([0.0, 0.0]))
print('With theta = [0, 0] \nCost computed = %.2f' % J)


# In[20]:


# further testing of the cost function
J = computeCost(X, y, theta=np.array([-1, 2]))
print('With theta = [-1, 2]\nCost computed = %.2f' % J)


# # Gradient Descent

# In[21]:


def gradientDescent(X, y, theta, alpha, num_iters):
   
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    
    J_history = [] # Use a python list to save cost in every iteration
    
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)

        # =====================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history


# In[22]:


# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))


# In[24]:


# plot the linear fit
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')
pyplot.legend(['Training data', 'Linear regression']);


# # Predict for new values
# 
# We give new inputs, and for that we check the output.

# In[26]:


# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))


# # For Contour plots - Refer Git

# # Linear regression with multiple variables

# In[27]:


data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
# input values are multiple.

y = data[:, 2]
# one output value - It is in 3rd column - which is second index.
m = y.size


# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


# # Feature Normalization:
# 
# One of the input columns has higer values. Normalize it to mean value.

# In[28]:


def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

  
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    
    return X_norm, mu, sigma


# In[29]:


# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# In[30]:


X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
# Should add extra column which has 1's - For theta 0.


# # Cost Function along with Gradient descent
# 
# Here cost function is called within gradient descent function.
# Theta values are initially taken as random.

# In[32]:


def computeCostMulti(X, y, theta):
    
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    h = np.dot(X, theta)
    
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    
    return J


# In[33]:


def gradientDescentMulti(X, y, theta, alpha, num_iters):
   
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
    

        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history


# In[36]:


# Choose some alpha value - change this
alpha = 0.1
num_iters = 400

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)


# In[37]:


# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))


# # Plot of cost function with no.of iterations
# 
# cost function values are in J history array.

# In[38]:


# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# # Prediction for new values.
# 
# Do normalization for new values also if we have done it initially.
# 

# In[39]:


X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)   # You should change this

# ===================================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))


# # Normal Equations - We use direct formula for prediction.
# 
# Using this formula does not require any feature scaling

# In[41]:


data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)


# In[42]:


def normalEqn(X, y):
   
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    
    return theta


# In[43]:


# Calculate the parameters from the normal equation
theta = normalEqn(X, y);


# In[48]:


# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta)));


# # Predict output for new values
# 
# Same 

# In[49]:


X_array = [1, 1650, 3]
# X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta) # You should change this

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
# 


# In[ ]:





# In[ ]:




