import numpy as np
from testCases import *
from planar_utils import plot_decision_boundary,load_planar_dataset,load_extra_datasets
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


def sigmoid(z):
    res = 1/(1+np.exp(-z))
    return res

def layer_size(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x,n_h,n_y

def initialize_param(n_x,n_h,n_y):
    w1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    w2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))

    assert(w1.shape == ( n_h , n_x ))
    assert(b1.shape == ( n_h , 1 ))
    assert(w2.shape == ( n_y , n_h ))
    assert(b2.shape == ( n_y , 1 ))

    params = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }
    return params

def forward_propagate(X,params):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]
    
    z1 = np.dot(w1,X)+b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = sigmoid(z2)

    A_Z = {
        "z1":z1,
        "a1":a1,
        "z2":z2,
        "a2":a2
    }
    return A_Z

def backward_propagate(X,Y,A_Z,params):
    m = X.shape[1]
    
    a1 = A_Z["a1"]
    a2 = A_Z["a2"]

    w2 = params["w2"]

    dz2 = a2 - Y
    dw2 = (1/m)*np.dot(dz2,a1.T)
    db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1 = np.multiply(np.dot(w2.T,dz2),1-np.power(a1,2))
    dw1 = (1/m)*np.dot(dz1,X.T)
    db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)

    grads = {
        "dw1":dw1,
        "db1":db1,
        "dw2":dw2,
        "db2":db2
    }

    return grads

def update_params(params,grads,learning_rate):
    w1,w2 = params["w1"],params["w2"]
    b1,b2 = params["b1"],params["b2"]

    dw1,dw2 = grads["dw1"],grads["dw2"]
    db1,db2 = grads["db1"],grads["db2"]

    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    b1 = db1 - learning_rate*db1
    b2 = db2 - learning_rate*db2

    params = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
    }

    return params

def cost_count(a2,Y):
    m = Y.shape[1]

    logprobs = np.multiply(np.log(a2),Y) + np.multiply((1-Y),np.log(1-a2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost))

    assert(isinstance(cost,float))

    return cost

def two_layer_NN(params,num_iterations,print_cost = True):
    for i in range(num_iterations):
        A_Z = forward_propagate(X,params)
        a2 = A_Z["a2"]
        cost = cost_count(a2,Y)
        grads = backward_propagate(X,Y,A_Z,params)
        params = update_params(params,grads,learning_rate = 0.5)

        if print_cost:
            if i%1000 == 0:
                print("the cost of the",i,"cicle is "+str(cost))
    
    return params

def predict(params,X):
    A_Z = forward_propagate(X,params)
    a2 = A_Z["a2"]

    prediction = np.round(a2)

    return prediction

if __name__ == "__main__":
    X,Y = load_planar_dataset()
    n_x,n_h,n_y = layer_size(X,Y)
    params = initialize_param(n_x,n_h,n_y)
    
    params = two_layer_NN(params,10000)

    prediction = predict(params,X)
    print('correct rate %d' % float((np.dot(Y,prediction.T)+np.dot(1-Y,1-prediction.T))/float(Y.size)*100)+'%')