from inflection import parameterize
import tensorflow as tf
import numpy as np
import h5py
import math
import cnn_utils
import tf_utils

def zero_pad(X,pad):
    X_paded = np.pad(X,(
        (0,0),       #样本数，不填充
        (pad,pad),   #图像高度,你可以视为上面填充x个，下面填充y个(x,y)
        (pad,pad),   #图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
        (0,0)),      #通道数，不填充
        'constant', constant_values=0)      #连续一样的值填充

    return X_paded

def conv_single_step(a_slice_prev,W,b):
    s = np.multiply(a_slice_prev,W)+b
    Z = np.sum(s)

    return Z

#卷积前向传播
def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev-f+2*pad)/stride)+1
    n_W = int((n_W_prev-f+2*pad)/stride)+1

    Z = np.zeros((m,n_H,n_W,n_C))

    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):                              #遍历样本
        a_prev_pad = A_prev_pad[i]                  #选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):                        #在输出的垂直轴上循环
            for w in range(n_W):                    #在输出的水平轴上循环
                for c in range(n_C):                #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride         #竖向，开始的位置
                    vert_end = vert_start + f       #竖向，结束的位置
                    horiz_start = w * stride        #横向，开始的位置
                    horiz_end = horiz_start + f     #横向，结束的位置
                    #切片位置定位好了我们就把它取出来,需要注意的是我们是“穿透”取出来的，
                    #自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    #执行单步卷积
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[: ,: ,: ,c],b[0,0,0,c])

    cache = (A_prev,W,b,hparameters)

    return (Z,cache)

#池化层前向传播
def pool_forward(A_prev,hparameters,mode = "max"):
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev-f)/stride)+1
    n_W = int((n_W_prev-f)/stride)+1
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):                              #遍历样本
        for h in range(n_H):                        #在输出的垂直轴上循环
            for w in range(n_W):                    #在输出的水平轴上循环
                for c in range(n_C):                #循环遍历输出的通道
                    #定位当前的切片位置
                    vert_start = h * stride         #竖向，开始的位置
                    vert_end = vert_start + f       #竖向，结束的位置
                    horiz_start = w * stride        #横向，开始的位置
                    horiz_end = horiz_start + f     #横向，结束的位置
                    #定位完毕，开始切割
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    #对切片进行池化操作
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice_prev)

    cache = (A_prev,hparameters)

    return A,cache

def conv_backward(dZ,cache):
    (A_prev,W,b,hparameters) = cache
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dZ.shape
    (f,f,n_C_prev,n_C) = W.shape

    pad = hparameters["pad"]
    stride = hparameters["stride"]

    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))    
    db = np.zeros((1,1,1,n_C))

    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)

    for i in range(m):
        #选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #定位切片位置
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f
                    
                    #定位完毕，开始切片
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    #切片完毕，使用上面的公式计算梯度
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
        #设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    return (dA_prev,dW,db)

#从输入矩阵中创建掩码，以保存最大值的矩阵的位置
#最大池化层反向传播
def create_mask_from_window(x):
    mask = x == np.max(x)

    return mask

#给定一个值，为按矩阵大小平均分配到每一个矩阵位置中
#均值池化层反向传播
def distribute_value(dz,shape):
    (n_H,n_W) = shape
    average = dz/(n_H*n_W)
    a = np.ones(shape)*average

    return a

def pool_backward(dA,cache,mode = "max"):
    (A_prev,hparameters) = cache

    f = hparameters["f"]
    stride = hparameters["stride"]

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dA.shape

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start+f
                    horiz_start = w
                    horiz_end = horiz_start+f

                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,shape)

    return dA_prev                    

#创建占位符
def create_placeholders(n_H0,n_W0,n_C0,n_y):
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return X,Y

#初始化参数
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.compat.v1.get_variable("W1",[4,4,3,8],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.compat.v1.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {
        "W1":W1,
        "W2":W2
    }

    return parameters

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #Conv2d : 步伐：1，填充方式：“SAME”
    Z1 = tf.nn.conv2d(X,W1,stride = [1,1,1,1],padding = "SAME")
    A1 = tf.nn.relu(Z1)
    #Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding="SAME")

    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],stride=[1,4,4,1],padding="SAME")

    #一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    #全连接层（FC）：使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn = None)

    return Z3

def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))

    return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.009,num_epochs = 100,minibatch_size = 64):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variable_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed += 1
            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _,temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                minibatch_cost += temp_cost/num_minibatches
            
            if epoch%5 == 0:
                print("the "+str(epoch)+"th's cost is "+str(minibatch_cost))
            if epoch%1 == 0:
                costs.append(minibatch_cost)
            
        predict_op = tf.argmax(Z3,1)
        corrent_prediction = tf.equal(predict_op,tf.argmax(Y,1))

        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
        print("corrent_prediction accuracy = "+str(accuracy))

        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})

        print("the accuracy of test set is "+str(test_accuracy))

        return (test_accuracy,parameters)

if __name__ == "__main__":
    X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = tf_utils.load_dataset()
    X_train = X_train_orig/255
    X_test = X_test_orig/255
    Y_train = cnn_utils.convert_to_one_hot(Y_train_orig,6).T
    Y_test = cnn_utils.convert_to_one_hot(Y_test_orig,6).T

    (test_accuracy,parameters) = model(X_train,Y_train,X_test,Y_test)
