import pandas as pd
from sklearn.model_selection import train_test_split
import math



data = pd.read_csv('data.csv')

data.pop('id')

data['diagnosis'] = data['diagnosis'].replace('M','-1')
data['diagnosis'] = data['diagnosis'].replace('B','1')


x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],data.iloc[:,0] , test_size=0.2 )

# Standardization
import numpy

# calculate mu and sig using the training set
d = x_train.shape[1]
mu = numpy.mean(x_train, axis=0).values.reshape(1, d)
sig = numpy.std(x_train, axis=0).values.reshape(1, d)

# transform the training features
x_train = (x_train - mu) / (sig + 1E-6)

# transform the test features
x_test = (x_test - mu) / (sig + 1E-6)

# print('test mean = ')
# print(numpy.mean(x_test, axis=0))

# print('test std = ')
# print(numpy.std(x_test, axis=0))

#LOG OR LOG10

# Calculate the objective function value, or loss
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     objective function value, or loss (scalar)
def objective(w, x, y, lam):
    sum = 0 #summation
    i = 1
    n=x.shape[0]
    while ( i < n): # from i = 1 to n 
        sum += math.log(1 + numpy.exp(-y[i] *numpy.dot( x[i].T, w)  ))
        i+=1
    sum /= n
    sum +=(lam/2)* (numpy.linalg.norm(w, 2)**2) 

    return sum
    
    
    
# Calculate the gradient
# Inputs:
#     w: weight: d-by-1 matrix
#     x: data: n-by-d matrix
#     y: label: n-by-1 matrix
#     lam: regularization parameter: scalar
# Return:
#     g: gradient: d-by-1 matrix

def gradient(w, x, y, lam):
    sum = 0
    i = 1
    n = x.shape[0]
    while i < n:
        sum += (y[i]* x[i])/(1 + math.exp(numpy.dot( x[i].T, w)*y[i]))
        i += 1
    # sum /= -n
    # (print((lam*w).shape))
    # print(sum.shape)
    
    # sum += (lam* w)
    return ((sum / -n) + (lam*w).T).T


w = numpy.zeros((29, 1))
x = numpy.zeros((455, 29))
y = numpy.zeros((455, 1))

print(gradient(w, x, y, 0.01).shape)
        
        

        
        
        
    
        
# Calculate the objective Q_i and the gradient of Q_i
# Inputs:
#     w: weights: d-by-1 matrix
#     xi: data: 1-by-d matrix
#     yi: label: scalar
#     lam: scalar, the regularization parameter
# Return:
#     obj: scalar, the objective Q_i
#     g: d-by-1 matrix, gradient of Q_i

def stochastic_objective_gradient(w, xi, yi, lam):
    grad = -(yi*xi)/(1+ numpy.exp(-yi *numpy.dot( xi.T, w))) + lam*w
    obj += math.log10(1 + numpy.exp(-yi *numpy.dot( xi.T, w)  ))
    obj +=(lam/2)* (numpy.linalg.norm(w, 2)**2) 
    return obj, grad