import numpy as np
import rnn_utils

#实现RNN单元的单步前向传播
def rnn_cell_forward(xt,a_prev,parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)

    yt_pred = rnn_utils.softmax(np.dot(Wya,a_next)+by)

    cache = (a_next,a_prev,xt,parameters)

    return a_next,yt_pred,cache

#实现循环神经网络的前向传播
def rnn_forward(x,a0,parameters):
    # 获取 x 与 Wya 的维度信息
    caches = []
    n_x,m,T_x = x.shape
    n_y,n_a = parameters["Wya"].shape

    # 使用0来初始化“a” 与“y”
    a = np.zeros([n_a,m,T_x])
    y_pred = np.zeros([n_y,m,T_x])

    # 初始化“next”
    a_next = a0

    # 遍历所有时间步
    for t in range(T_x):
        ## 1.使用rnn_cell_forward函数来更新“next”隐藏状态与cache。
        a_next,yt_pred,cache = rnn_cell_forward(x[:,:,t],a_next,parameters)
        ## 2.使用 a 来保存“next”隐藏状态（第 t ）个位置。
        a[:,:,t] = a_next
        ## 3.使用 y 来保存预测值。
        y_pred[:,:,t] = yt_pred
        ## 4.把cache保存到“caches”列表中。
        caches.append(cache)

    caches = (caches,x)

    return a,y_pred,caches

#实现一个LSTM单元的前向传播
def lstm_cell_forward(xt,a_prev,c_prev,parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x,m = xt.shape
    n_y,n_a = Wy.shape

    # 1.连接 a_prev 与 xt
    contact = np.zeros([n_a+n_x,m])
    contact[:n_a,:] = a_prev
    contact[n_a:,:] = xt

    # 2.根据公式计算ft、it、cct、c_next、ot、a_next
    ## 遗忘门
    ft = rnn_utils.sigmoid(np.dot(Wf,contact)+bf)

    ## 更新门
    it = rnn_utils.sigmoid(np.dot(Wi,contact)+bi)

    ## 更新单元
    cct = np.tanh(np.dot(Wc,contact)+bc)

    ## 更新单元
    c_next = ft*c_prev+it*cct

    ## 输出门
    ot = rnn_utils.sigmoid(np.dot(Wo,contact)+bo)
    a_next = ot*np.tanh(c_next)

    # 3.计算LSTM单元的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy,a_next)+by)

    # 保存包含了反向传播所需要的参数
    cache = (a_next,c_next,a_prev,c_prev,ft,it,cct,ot,xt,parameters)

    return a_next,c_next,yt_pred,cache

#实现LSTM单元组成的的循环神经网络
def lstm_forward(x,a0,parameters):
    caches = []
    n_x,m,T_x = x.shape
    n_y,n_a = parameters["Wy"].shape

    # 使用0来初始化“a”、“c”、“y”
    a = np.zeros([n_a,m,T_x])
    c = np.zeros([n_a,m,T_x])
    y = np.zeros([n_y,m,T_x])

    # 初始化“a_next”、“c_next”
    a_next = a0
    c_next = np.zeros([n_a,m])

    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
        a_next,c_next,yt_pred,cache = lstm_cell_forward(x[:,:,t],a_next,c_next,parameters)
        # 保存新的下一个隐藏状态到变量a中
        a[:,:,t] = a_next
        # 保存预测值到变量y中
        y[:,:,t] = yt_pred
        # 保存下一个单元状态到变量c中
        c[:,:,t] = c_next
        # 把cache添加到caches中
        caches.append(cache)

    caches = (cache,x)

    return a,y,c,caches