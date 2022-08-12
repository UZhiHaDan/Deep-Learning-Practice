from argon2 import Parameters
from inflection import parameterize
import numpy as np
import random
import time
import cllm_utils

# 梯度修剪
def clip(gradients,maxValue):
    # 获取参数
    dWaa,dWax,dWya,db,dby = gradients['dWaa'],gradients['dWax'],gradients['dWya'],gradients['db'],gradients['dby']

    for gradient in [dWaa,dWax,dWya,db,dby]:
        np.clip(gradient,-maxValue,maxValue,out=gradient)
    
    return gradients

# 采样
def sample(parameters,char_to_is,seed):
    # 从parameters 中获取参数
    Waa,Wax,Wya,by,b = parameters['Waa'],parameters['Wax'],parameters['Wya'],parameters['by'],parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 步骤1 
    ## 创建独热向量
    x = np.zeros((vocab_size,1))

    ## 使用0初始化a_prev
    a_prev = np.zeros((n_a,1))

    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []

    # IDX是检测换行符的标志，我们将其初始化为-1
    idx = -1

    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    #（我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    counter = 0
    newline_character = char_to_ix

    while(idx != newline_character and counter<50):
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z = np.dot(Wya,a)+by
        y = cllm_utils.softmax(z)

        # 设定随机种子
        np.random.seed(counter + seed)

        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)),p=y.ravel())

        # 添加到索引中
        indices.append(idx)

        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size,1))
        x[idx] = 1

        # 更新a_prev为a
        a_prev = a 

        # 累加器
        seed += 1
        counter +=1

        if(counter == 50):
            indices.append(char_to_ix["\n"])

        return indices
# 执行训练模型的单步优化。
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)
    
    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)
    
    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients,5)
    
    # 更新参数
    parameters = cllm_utils.update_parameters(parameters,gradients,learning_rate)
    
    return loss, gradients, a[len(X)-1]
    
# 训练模型并形成恐龙名字
def model(data,ix_to_char,char_to_ix,num_iterations=3500,n_a=50,dino_name=7,vocab_size=27):
    # 从vocab_size中获取n_x、n_y
    n_x,n_y = vocab_size,vocab_size
    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a,n_x,n_y)
    # 初始化损失
    loss = cllm_utils.get_initial_loss(vocab_size,dino_name)
    # 构建恐龙名称列表
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

     # 初始化LSTM隐藏状态
     a_prev = np.zeros((n_a,1))

     for j in random(num_iterations):
        # 定义一个训练样本
        index = j%len(examples)
        X = [None]+[char_to_ix[ch] for ch in examples[indexs]]
        Y = X[1:]+[char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        # 选择学习率为0.01
        curr_loss,gradients,a_prev = optimize(X,Y,a_prev,parameters)
        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss,curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j%200 == 0:
            print("the "+str(j+1)+"th loss is"+str(loss))

            seed = 0
            for name in range(dino_name):
                # 采样
                sampled_indices = sample(parameters,char_to_ix,seed)
                cllm_utils.print_sample(sampled_indices,ix_to_char)
                seed += 1
            print("\n")
    return parameters

if __name__ == "__main__":
    data = open("dinos.txt","r").read()

    # 转化为小写字符
    data = data.lower()

    # 转化为无序且不重复的元素列表
    chars = list(set(data))

    # 获取大小信息
    data_size,vocab_size = len(data),len(chars)

    # 创建字典
    char_to_ix = {ch:i for i,ch in enumerate(sorted(chars))}
    ix_to_char = {i:ch for i,ch in enumerate(sorted(chars))}

