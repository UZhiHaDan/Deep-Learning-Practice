from keras.models import Sequential
from keras.layers import Conv2D,ZeroPadding2D,Activation,Input,Concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D,AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda,Flatten,Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K

from IPython.display import SVG 
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

K.set_image_data_format('channels_first')

import time
import cv2
import os
import numpy as np
from numpy import genfromtxt, identity, positive
import pandas as pd
import tensorflow as tf
import fr_utils
from inception_blocks_v2 import *

#根据公式实现三元组损失函数
def triplet_loss(y_true,y_pred,alpha=0.2):
    #获取anchor, positive, negative的图像编码
    anchor,positice,negative = y_pred[0],y_pred[1],y_pred[2]

    #第一步：计算"anchor" 与 "positive"之间编码的距离，这里需要使用axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)

    #第二步：计算"anchor" 与 "negative"之间编码的距离，这里需要使用axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)

    #第三步：减去之前的两个距离，然后加上alpha
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)

    #通过取带零的最大值和对训练样本的求和来计算整个公式
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))

    return loss

#对“identity”与“image_path”的编码进行验证。
def verify(image_path,identity,database,model):
    #第一步：计算图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path,model)

    #第二步：计算与数据库中保存的编码的差距
    dist = np.linalg.norm(encoding-database[identity])

    #第三步：判断是否打开门
    if dist<0.7:
        print("welcome "+str(identity))
        is_door_open = True
    else:
        print("Sorry "+str(identity))
        is_door_open = False

    return dist,is_door_open

#根据指定的图片来进行人脸识别
def who_is_it(image_path,database,model):
    #步骤1：计算指定图像的编码，使用fr_utils.img_to_encoding()来计算。
    encoding = fr_utils.img_to_encoding(image_path,model)

    #步骤2 ：找到最相近的编码
    ## 初始化min_dist变量为足够大的数字，这里设置为100
    min_dist = 100

    ## 遍历数据库找到最相近的编码
    for (name,db_enc) in database.items():
        ### 计算目标编码与当前数据库编码之间的L2差距。
        dist = np.linalg.norm(encoding-db_enc)

        ### 如果差距小于min_dist，那么就更新名字与编码到identity与min_dist中。
        if dist<min_dist:
            min_dist = dist
            identity = name
    
    if min_dist>0.7:
        print("sorry")
    else:
        print("name "+str(identity)+"distence "+str(min_dist))

    return min_dist,identity

if __name__ == "__main":
    #创建一个人脸识别的模型
    FRmodel = faceRecoModel(input_shape=(3,96,96))

    #加载训练好了的模型
    start_time = time.clock()
    FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
    fr_utils.load_weights_from_FaceNet(FRmodel)
    end_time = time.clock()
    minium = end_time-start_time
    print("using "+str(int(minium/60))+" mins"+str(int(minium%60))+"seconds")

    #加载数据库
    database = {}
    database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FRmodel)
    database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FRmodel)
    database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FRmodel)
    database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FRmodel)
    database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FRmodel)
    database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FRmodel)
    database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FRmodel)
    database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FRmodel)
    database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FRmodel)
    database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FRmodel)
    database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FRmodel)
    database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FRmodel)

    #验证
    verify("image_path","person_name",database,FRmodel)