from statistics import mode
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import IPython
import sys
from music21 import *
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from sympy import Mod

# 实现模型
def djmodel(Tx,n_a,n_values):
    # 定义输入数据的维度
    X = Input((Tx,n_values))

    # 定义a0, 初始化隐藏状态
    a0 = Input(shape=(n_a,),name="a0")
    c0 = Input(shape=(n_a,),name="c0")
    a = a0
    c = c0

    # 第一步：创建一个空的outputs列表来保存LSTM的所有时间步的输出。
    outputs = []

    for t in range(Tx):
        ## 2.A：从X中选择第“t”个时间步向量
        x = Lambda(lambda x:X[:,t,:])(X)
        ## 2.B：使用reshapor来对x进行重构为(1, n_values)
        x = reshapor(x)
        ## 2.C：单步传播
        a,_,c = LSTM_cell(x,initial_state=[a,c])
        ## 2.D：使用densor()应用于LSTM_Cell的隐藏状态输出
        out = densor(a)
        ## 2.E：把预测值添加到"outputs"列表中
        outputs.append(out)

    # 第三步：创建模型实体
    model = Model(input=[X,a0,c0],outputs=outputs)

    return model

def music_inference_model(LSTM_cell,densor,n_values=78,n_a=64,Ty=100):
    # 定义模型输入的维度
    x0 = Input(shape=(1,n_values))
    # 定义s0，初始化隐藏状态
    a0 = Input(shape=(n_a,),name="a0")
    c0 = Input(shape=(n_a,),name="c0")
    a = a0
    c = c0
    x = x0

    # 步骤1：创建一个空的outputs列表来保存预测值。
    outputs = []
    # 步骤2：遍历Ty，生成所有时间步的输出
    for t in range(Ty):
        # 步骤2.A：在LSTM中单步传播
        a,_,c = LSTM_cell(x,initial_state=[a,c])
        # 步骤2.B：使用densor()应用于LSTM_Cell的隐藏状态输出
        out = densor(a)
        # 步骤2.C：预测值添加到"outputs"列表中
        outputs.append(out)
        # 根据“out”选择下一个值，并将“x”设置为所选值的一个独热编码，
        # 该值将在下一步作为输入传递给LSTM_cell。我们已经提供了执行此操作所需的代码
        x = Lambda(one_hot)(out)

    inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)

    return inference_model

# 使用模型预处当前值的下一个值
def predict_and_sample(inference_model,x_initializer=x_initializer,a_initializer=a_initializer,c_initializer=c_initializer):
    # 步骤1：模型来预测给定x_initializer, a_initializer and c_initializer的输出序列
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    
    # 步骤2：将“pred”转换为具有最大概率的索引数组np.array()。
    indices = np.argmax(pred, axis=-1)
    
    # 步骤3：将索引转换为它们的一个独热编码。
    results = to_categorical(indices, num_classes=78)
    
    return results, indices

if __name__ == "__main__":
    IPython.display.Audio("D:/deep learning course/Course5/week1/data/30s_seq.mp3")

    X,Y,n_values,indices_values = load_music_utils()

    n_a = 64
    reshapor = Reshape((1, 78))                        #2.B
    LSTM_cell = LSTM(n_a, return_state = True)        #2.C
    densor = Dense(n_values, activation='softmax')    #2.D
    x = Lambda(lambda x: X[:,t,:])(X)
    a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])
