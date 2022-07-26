import numpy as np
import sklearn
import sklearn.datasets
import init_utils

def initialize_parameters_zeros(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for i in range(1,L):
        parameters["W"+str(i)] = np.zeros((layer_dims[i],layer_dims[i-1]))
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))

    return parameters

def initialize_parameters_random(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*10
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
    
    return parameters

def initialize_parameters_he(lay_dims):
    parameters = {}
    L = len(lay_dims)

    for i in range(1,L):
        parameters["W"+str(i)] = np.random.randn(lay_dims[i],lay_dims[i-1])*np.sqrt(2/lay_dims[i-1])
        parameters["b"+str(i)] = np.zeros((lay_dims[i],1))

    return parameters

def model(X,Y,initialization,learning_rate = 0.01,num_iteration = 15000):
    grads = {}
    costs = []
    layer_dims = [X.shape[0],10,5,1]

    if initialization == "zero":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    else:
        parameters = initialize_parameters_he(layer_dims)
    
    for i in range(0,num_iteration):
        A3,cache = init_utils.forward_propagation(X,parameters)
        cost = init_utils.compute_loss(A3,Y)
        grads = init_utils.backward_propagation(X,Y,cache)
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        if i%1000 == 0:
            costs.append(cost)
            print(str(i)+"th cicle's cost is "+str(cost))
    
    return parameters

if __name__ == "__main__":
    train_x,train_y,test_x,test_y = init_utils.load_dataset(is_plot=False)
    # 初始化为0
    # parameters = model(train_x,train_y,"zero")
    
    # 随机初始化
    # parameters = model(train_x,train_y,"random")

    # 抑梯度异常初始化
    parameters = model(train_x,train_y,"he")
    prediction = init_utils.predict(test_x,test_y,parameters)