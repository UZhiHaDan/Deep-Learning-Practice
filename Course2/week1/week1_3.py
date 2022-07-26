from functools import cache
import gc
from re import A
from tkinter import W
from tkinter.tix import Tree
import numpy as np
import sklearn
import gc_utils

# 一维线性
def forward_propagation(X,theta):
    J = np.dot(theta,X)

    return J

def back_propagation(X,theta):
    dtheta = X

    return dtheta

def gradient_check(X,theta,epsilon = 1e-7):
    thetaplus = theta+epsilon
    thetaminus = theta-epsilon
    J_plus = forward_propagation(X,thetaplus)
    J_minus = forward_propagation(X,thetaminus)
    gradapprox = (J_plus-J_minus)/(2*epsilon)

    grad = back_propagation(X,theta)

    numerator = np.linalg.norm(grad-gradapprox) # 求范数,linalg本意为linear+algebra,norm表示范数
    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    diff = numerator/denominator

    if diff<1e-7:
        print("Correct!")
    else:
        print("Wrong")
    
    return diff

# 高维
def forward_propagation_n(X,Y,parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X)+b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W,A1)+b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3,A2)+b3
    A3 = gc_utils.sigmoid(Z3)

    logprobs = np.multiply(-np.log(A3),Y)+np.multiply(np.log(1-A3),1-Y)
    cost = np.sum(logprobs)/m

    cache = (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)

    return cost,cache

def backward_propagation_n(X,Y,cache):
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    m = X.shape[1]
    grads = {}

    dZ3 = A3-Y
    dW3 = np.dot(dZ3,A2.T)/m
    db3 = np.sum(dZ3,axis=1,keepdims=True)/m

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=Tree)/m

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m

    grads = {
        "dZ3": dZ3, "dW3": dW3, "db3": db3,
        "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
        "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1
    }

    return grads

def gradient_check_n(parameters,gradients,X,Y,epsilon=1e-7):
    parameters_values,keys = gc_utils.dictionary_to_vector(parameters)
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))

    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        J_plus[i],cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaplus))

        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon
        J_minus[i],cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaminus))

        gradapprox[i] = (J_plus-J_minus)/(2*epsilon)
    
    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    diff = numerator/denominator

    if diff<1e-7:
        print("correct")
    else:
        print("Wrong")
    
    return diff

if __name__ == "__main__":
    diff = gradient_check(2,4)