from ctypes.wintypes import WIN32_FIND_DATAW
import numpy as np
import sklearn
import reg_utils
'''
正则化与dropout的使用
'''
# dropout
def forward_propagation_with_dropout(X,parameters,keep_prob):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X)+b1
    A1 = reg_utils.relu(Z1)

    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1<keep_prob
    A1 = (A1*D1)/keep_prob

    Z2 = np.dot(W2,A1)+b2
    A2 = reg_utils.relu(Z2)

    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2<keep_prob
    A2 = (A2*D2)/keep_prob

    Z3 = np.dot(W3,A2)+b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3-Y
    dW3 = np.dot(dZ3,A2.T)/m
    db3 = np.sum(dZ3,axis=1,keepdims=True)/m
    dA2 = np.dot(W3.T,dZ3)
    dA2 = (dA2*D2)/keep_prob

    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dA1 = np.dot(W2.T,dZ2)
    dA1 = (dA1*D1)/keep_prob

    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m

    grads = {
        "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
        "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
        "dZ1":dZ1,"dW1":dW1,"db1":db1
    }

    return grads

# 正则化
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3,Y)

    L2_retularization_cost = lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m)

    cost = cross_entropy_cost+L2_retularization_cost

    return cost

def backward_propagation_with_regularization(X,Y,cache,lambd):
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3-Y

    dW3 = np.dot(dZ3,A2.T)/m+(lambd*W3)/m
    db3 = np.sum(dZ3,axis=1,keepdims=True)/m

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = np.dot(dZ2,A1.T)/m+(lambd*W2)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = np.dot(dZ1,X.T)/m+(lambd*W1)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m

    grads = {
        "dZ3":dZ3,"dW3":dW3,"db3":db3,"dA2":dA2,
        "dZ2":dZ2,"dW2":dW2,"db2":db2,"dA1":dA1,
        "dZ1":dZ1,"dW1":dW1,"db1":db1
    }

    return grads

def model(X,Y,lambd,keep_prob,learning_rate = 0.3,num_iteration = 30000):
    grads = {}
    costs = []
    layer_dims = [X.shape[0],20,3,1]

    parameters = reg_utils.initialize_parameters(layer_dims)

    for i in range(0,num_iteration):
        if keep_prob == 1:
            A3,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            A3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        
        if lambd == 0:
            cost = reg_utils.compute_cost(A3,Y)
        else:
            cost = compute_cost_with_regularization(A3,Y,parameters,lambd)
        
        if(lambd == 0 and keep_prob == 1):
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)

        if i%10000 == 0:
            costs.append(cost)
            print(str(i)+"th circle's cost is "+str(cost))

    return parameters

if __name__ == "__main__":
    train_x,train_y,test_x,test_y = reg_utils.load_2D_dataset(is_plot=False)
    # 没有正则化
    # parameters = model(train_x,train_y,0,1)

    # 正则化
    # parameters = model(train_x,train_y,0.7,1)

    # dropout
    parameters = model(train_x,train_y,0,0.86)
    prediction = reg_utils.predict(test_x,test_y,parameters)