'''
构建能接受文字序列的模型
这个模型会考虑文字顺序 使用已经训练好的词嵌入
'''
from random import shuffle
import numpy as np
np.random.seed(0)
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(1)
from keras.initializers import glorot_uniform

def sentences_to_indices(X,word_to_index,max_len):
    #输入字符串类型的句子的数组，转化为对应的句子列表
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))

    for i in range(m):
        sentences_words = X[i].lower().split()
        j = 0
        
        for w in sentences_words:
            X_indices[i,j] = word_to_index[w]
            j += 1

    return X_indices

def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    vocab_len = len(word_to_index)+1
    emb_dim = word_to_vec_map["cucumber"].shape[0]

    emb_matrix = np.zeros((vocab_len,emb_dim))

    for word,index in word_to_index.items():
        emb_matrix[index,:] = word_to_vec_map[word]
    
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojify_v2(input_shape,word_to_vec_map,word_to_index):
    sentences_indices = Input(input_shape,dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    embeddings = embedding_layer(sentences_indices)
    X = LSTM(128,return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128,return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentences_indices,outputs=X)

    return model

if __name__ == "__main__":
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
    X_trian_indices = sentences_to_indices(X_train,word_to_index,maxLen)
    Y_train_oh = Emojify_v2.convert_to_one_hot(Y_trian,C=5)
    model.fit(X_trian_indices,Y_train_oh,epochs=50,batch_size=32,shuffle=True)
    Y_test_oh
