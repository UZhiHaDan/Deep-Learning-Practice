from multiprocessing import reduction
import numpy as np
import emo_utils
import emoji

#将句子转换为单词列表，提取其GloVe向量，然后将其平均。
def sentence_to_avg(sentence,word_to_vec_map):
    #分割句子
    words = sentence.lower().split()
    #初始化均值词向量
    avg = np.zeros(50,)
    #对词向量取平均值
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg,len(words))

    return avg

#在numpy中训练词向量模型
def model(X,Y,word_to_vec_map,learning_rate=0.1,num_iteration=400):
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    #初始化参数
    W = np.random.randn(n_y,n_h)/np.sqrt(n_h)
    b = np.zeros((n_y,))

    #将Y转为独热编码
    Y_oh = emo_utils.convert_to_one_hot(Y,C=n_y)

    for t in range(num_iteration):
        for i in range(m):
            #获取第i个训练样本均值
            avg = sentence_to_avg(X[i],word_to_vec_map)

            #前向传播
            z = np.dot(W,avg)+b
            a = emo_utils.softmax(z)

            #计算第i个训练样本的损失
            cost = -np.sum(Y_oh[i]*np.log(a))

            #计算梯度
            dz = a-Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1),avg.reshape(1,n_h))
            db = dz

            #更新参数
            W = W-learning_rate*dW
            b = b=learning_rate*db
        if t%100 == 0:
            print("the "+str(t)+"th iteration's cost is "+str(cost))
            pred = emo_utils.predict(X,Y,W,b,word_to_vec_map)

    return pred,W,b    

if __name__ == "__main__":
    X_train,Y_train = emo_utils.read_csv("D:/deep learning course/Course5/week2/data/train_emoji.csv")
    X_test,Y_test = emo_utils.read_csv("D:/deep learning course/Course5/week2/data/test.csv")

    maxLen = len(max(X_train,key=len).split())

    #将标签转为softmax需要的格式
    Y_oh_train = emo_utils.convert_to_one_hot(Y_train,C=5)
    Y_oh_test = emo_utils.convert_to_one_hot(Y_test,C=5)

    word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs("D:/deep learning course/Course5/week2/data/glove.6B.50d.txt")

    pred,W,b = model(X_train,Y_train,word_to_vec_map)

    pred_test = emo_utils.predict(X_test,Y_test,W,b,word_to_vec_map)