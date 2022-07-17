import numpy as np
import matplotlib.pyplot as plt 
import h5py
from lr_utils import load_dataset

def propagate(w,b,train_x,train_y):
    m = train_x.shape[1]

    z = np.dot(w.T,train_x)+b
    a = sigomoid(z)

    y_hat = a-train_y

    cost = (-1.0/m)*np.sum(train_y*np.log(a)+(1-train_y)*np.log(1-a))
    dw = (1.0/m)*np.dot(train_x,y_hat.T)
    db = (1.0/m)*np.sum(y_hat)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
        "dw":dw,
        "db":db
    }

    return grads,cost

def optimize(w,b,train_x,train_y,num_iteration,learning_rate,print_cost = True):
    costs = []
    for i in range(num_iteration):
        grads,cost = propagate(w,b,train_x,train_y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        
        if (print_cost) and (i%100 == 0):
            print("cicle_time:%i,error_rate:%f" % (i,cost))
        
    params = {
        "w":w,
        "b":b
    }

    grads ={
        "dw":dw,
        "db":db
    }

    return params,grads,costs

def predict(w,b,X):
    m = X.shape[1] # m=209

    predition_y = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigomoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        predition_y[0,i] = 1 if A[0,i]>0.5 else 0

    assert(predition_y.shape == (1,m))

    return predition_y

def sigomoid(z):
    res = 1/(1+np.exp(-z))
    return res

def logistic():
    train_x,train_y,test_x,test_y,classes = load_dataset()

    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1]*train_x.shape[2]*train_x.shape[3]).T
    # (12288,209)
    test_x = test_x.reshape(test_x.shape[0],test_x.shape[1]*test_x.shape[2]*test_x.shape[3]).T
    # (12288,50)

    train_x = train_x/255
    test_x = test_x/255

    w = np.zeros((train_x.shape[0],1))
    b = 0

    parameters,gards,costs = optimize(w,b,train_x,train_y,2000,0.5)

    w,b = parameters["w"],parameters["b"]

    prediction_test = predict(w,b,test_x)
    prediction_train = predict(w,b,train_x)

    print("train correct:"  , format(100 - np.mean(np.abs(prediction_train - train_y)) * 100) ,"%")
    print("test correct:"  , format(100 - np.mean(np.abs(prediction_test - test_y)) * 100) ,"%")

    d = {
            "costs" : costs,
            "Y_prediction_test" : prediction_test,
            "Y_prediciton_train" : prediction_train,
            "w" : w,
            "b" : b
    }
    return d

if __name__ == '__main__':
    d = logistic()
