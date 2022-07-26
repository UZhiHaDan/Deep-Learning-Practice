import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import math
import sklearn 
import sklearn.datasets

import opt_utils
import testCase

# mini_batch梯度下降
def random_mini_batches(X,Y,mini_batch_size = 64):
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:,permutation]#将每一列的数据按permutation的顺序来重新排列
    shuffled_Y = Y[:,permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m%mini_batch_size != 0:
        #获取最后剩余的部分
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# 动量梯度下降
def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for i in range(L):
        v["dW"+str(i+1)] = np.zeros_like(parameters["W"+str(i+1)])# 输出（）中形状相同的列表，不同的是其中的元素都为0.
        v["db"+str(i+1)] = np.zeros_like(parameters["b"+str(i+1)])

    return v

def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    L = len(parameters)//2
    
    for i in range(L):
        v["dW"+str(i+1)] = beta*v["dW"+str(i+1)]+(1-beta)*grads["dW"+str(i+1)]
        v["db"+str(i+1)] = beta*v["db"+str(i+1)]+(1-beta)*grads["db"+str(i+1)]

        parameters["W"+str(i+1)] -= learning_rate*v["dW"+str(i+1)]
        parameters["b"+str(i+1)] -= learning_rate*v["db"+str(i+1)]
    
    return parameters,v

# Adam
def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for i in range(L):
        v["dW"+str(i+1)] = np.zeros_like(parameters["W"+str(i+1)])
        v["db"+str(i+1)] = np.zeros_like(parameters["b"+str(i+1)])

        s["dW"+str(i+1)] = np.zeros_like(parameters["W"+str(i+1)])
        s["db"+str(i+1)] = np.zeros_like(parameters["b"+str(i+1)])

    return (v,s)

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for i in range(L):
        v["dW"+str(i+1)] = beta1*v["dW"+str(i+1)]+(1-beta1)*grads["dW"+str(i+1)]
        v["dW"+str(i+1)] = beta1*v["db"+str(i+1)]+(1-beta1)*grads["db"+str(i+1)]
        v_corrected["dW"+str(i+1)] = v["dW"+str(i+1)]/(1-np.power(beta1,t))
        v_corrected["db"+str(i+1)] = v["db"+str(i+1)]/(1-np.power(beta1,t))

        s["dW"+str(i+1)] = beta2*s["dW"+str(i+1)]+(1-beta2)*np.square(grads["dW"+str(i+1)])
        s["db"+str(i+1)] = beta2*s["db"+str(i+1)]+(1-beta2)*np.square(grads["db"+str(i+1)])
        s_corrected["dW"+str(i+1)] = s["dW"+str(i+1)]/(1-np.power(beta2,t)) 
        s_corrected["db"+str(i+1)] = s["db"+str(i+1)]/(1-np.power(beta2,t))

        parameters["W"+str(i+1)] -= learning_rate*(v_corrected["dW"+str(i+1)]/np.sqrt(s_corrected["dW"+str(i+1)]+epsilon))
        parameters["b"+str(i+1)] -= learning_rate*(v_corrected["db"+str(i+1)]/np.sqrt(s_corrected["db"+str(i+1)]+epsilon))

    return (parameters,v,s)

def model(X,Y,layer_dims,optimizer,learning_rate = 0.0007,mini_batch_size = 64,beta = 0.9,beta1 = 0.9,beta2 = 0.999,epsilon = 1e-8,num_epochs = 10000):
    L = len(layer_dims)
    costs = []
    t = 0

    parameters = opt_utils.initialize_parameters(layer_dims)

    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v,s = initialize_adam(parameters)

    
    for i in range(num_epochs):
        minibatches = random_mini_batches(X,Y,mini_batch_size)

        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch

            A3,cache = opt_utils.forward_propagation(minibatch_X,parameters)
            cost = opt_utils.compute_cost(A3,minibatch_Y)
            grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)

            if optimizer == "momentum":
                parameters,v = update_parameters_with_momentun(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        
        if i%1000 == 0:
            costs.append(cost)
            print(str(i)+"th iteration's error rate is "+str(cost))
        
    return parameters


if __name__ == "__main__":
    train_X,train_Y = opt_utils.load_dataset(is_plot=True)
    layer_dims = [train_X.shape[0],5,2,1]
    parameters = model(train_X,train_Y,layer_dims,beta = 0.9,optimizer="momentum")
    # parameters = model(train_X,train_Y,layer_dims,optimizer="adam")

    prediction = opt_utils.predict(train_X,train_Y,parameters)
