import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
import lr_utils
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward

def layer_size(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x,n_h,n_y

def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    params = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return params

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])/np.sqrt(layer_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))

    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prew,W,b,activation):
    Z,linear_cache = linear_forward(A_prew,W,b) # linear_cache = (A,W,b)

    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z) # activation_cache = Z
    elif activation == "relu":
        A,activation_cache = relu(Z) # activation_cache = Z
    
    cache = (linear_cache,activation_cache) # cache = (A,W,b,Z)
    return A,cache

def L_model_forward(X,params):
    caches = []
    A = X
    L = len(params)//2  # 地板除，先做除法，再向下取整
    for i in range(1,L):
        A_prew = A
        A,cache = linear_activation_forward(A_prew,params['W'+str(i)],params['b'+str(i)],"relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A,params['W'+str(L)],params['b'+str(L)],"sigmoid")
    caches.append(cache)

    return AL,caches

def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m
    cost = np.squeeze(cost)

    return cost

def linear_backward(dZ,cache):
    A_prew,W,b = cache
    m = A_prew.shape[1]
    dW = np.dot(dZ,A_prew.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)

    dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

def L_model_backward(AL,Y,cache):
    grads = {}
    L = len(cache)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache = cache[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache,"sigmoid")

    for i in reversed(range(L-1)):
        current_cache = cache[i]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(i+2)],current_cache,"relu")
        grads["dA"+str(i+1)] = dA_prev_temp
        grads["dW"+str(i+1)] = dW_temp
        grads["db"+str(i+1)] = db_temp
    
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters)//2
    for i in range(L):
        parameters["W"+str(i+1)] = parameters["W"+str(i+1)]-learning_rate*grads["dW"+str(i+1)]
        parameters["b"+str(i+1)] = parameters["b"+str(i+1)]-learning_rate*grads["db"+str(i+1)]
    
    return parameters

def two_layer_model(X,Y,layer_dims,learning_rate = 0.0075,num_iterations = 2500,print_cost = False):
    grads = {}
    costs = []
    (n_x,n_h,n_y) = layer_dims

    params = initialize_parameters(n_x,n_h,n_y)

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    for i in range(0,num_iterations):
        A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")

        cost = compute_cost(A2,Y)

        dA2 = -(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        params = update_parameters(params,grads,learning_rate)

        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]

        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print(i,"th circle's cost is ",np.squeeze(cost))
    
    return params

def predict(X,Y,parameters):
    m = X.shape[1]
    p = np.zeros((1,m))

    probas,caches = L_model_forward(X,parameters)
    
    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    print("the correct rate is "+str(float(np.sum((p==Y))/m)))

    return p

def two_layer(train_x,train_y,test_x,test_y):
    n_x,n_h,n_y = layer_size(train_x,train_y)
    layer_dims = (n_x,n_h,n_y)

    parameters = two_layer_model(train_x,train_y,layer_dims)
    predictions_train = predict(train_x,train_y,parameters)
    predictions_test = predict(test_x,test_y,parameters)

def L_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iteration = 2500,print_cost = False):
    costs = []
    params = initialize_parameters_deep(layers_dims)
    for i in range(0,num_iteration):
        AL,caches = L_model_forward(X,params)
        cost = compute_cost(AL,Y)
        grads = L_model_backward(AL,Y,caches)
        params = update_parameters(params,grads,learning_rate)

        if i%100 == 0:
            costs.append(cost)
            if print_cost:
                print(i,"th cost is ",np.squeeze(cost))

    return params

def L_layer(train_x,train_y,test_x,test_y):
    layers_dims = [12288,20,7,5,1]
    params = L_layer_model(train_x,train_y,layers_dims) 

    pred_train = predict(train_x,train_y,params)
    pred_test = predict(test_x,test_y,params)

if __name__ == "__main__":
    train_set_X,train_set_y,test_set_x,test_set_y,classes = lr_utils.load_dataset()

    train_x_flatten = train_set_X.reshape(train_set_X.shape[0],-1).T
    test_x_flatten = test_set_x.reshape(test_set_x.shape[0],-1).T

    train_x = train_x_flatten/255
    train_y = train_set_y
    test_x = test_x_flatten/255
    test_y = test_set_y

    # two_layer(train_x,train_y,test_x,test_y)

    L_layer(train_x,train_y,test_x,test_y)
