from pyexpat import model
from statistics import mode
from turtle import st

from matplotlib.cbook import flatten
from regex import F
import numpy as np
from sympy import Add, DenseNDimArray, convolution
import tensorflow as tf
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPool2D,GlobalAvgPool2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

# import pydot 
from IPython.display import SVG
import scipy.misc
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import resnets_utils

def convolutional_block(X,f,filters,stage,block,s=2):
    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"
    
    #获取过滤器数量
    F1, F2, F3 = filters
    
    #保存输入数据
    X_shortcut = X
    
    #主路径
    ##主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)
    
    ##主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
    
    ##主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    
    #捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)
    
    #最后一步
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)
    
    return X

def identity_block(X,f,filters,stage,block):
    #定义命名规则
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"

    #获取过滤器
    F1,F2,F3 = filters

    #保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X

    #主路径的第一部分
    ##卷积层
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="vaild",
    name=conv_name_base+"2a",kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    #主路径的第二部分
    ##卷积层
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
    name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)

    #主路径的第三部分
    ##卷积层
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
    name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)

    #最后一步：
    ##将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    ##使用ReLU激活函数
    X = Activation("relu")(X)

    return X

def ResNets50(input_shape=(64,64,3),classes=6):
    #定义tensor类型的输入数据
    X_input = Input(input_shape)
    #0填充
    X = ZeroPadding2D((3,3))(X_input)

    #stage1
    X = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),name="conv1",
    kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=(3,3),strides=(2,2))(X)

    #stage2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block="a",s=1)
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="b")
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="c")

    #stage3
    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block="a",s=2)
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="b")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="c")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="d")

    #stage4
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="b")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="c")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="d")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="e")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="f")

    #stage5
    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="b")
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="c")

    #均值池化层
    X = AveragePooling2D(pool_size=(2,2),padding="same")(X)

    #输出层
    X = Flatten()(X)
    X = Dense(classes,activation="softmax",name="fc"+str(classes),
        kernel_initializer=glorot_uniform(seed=0))(X)
    
    #创建模型
    model = Model(input=X_input,outputs=X,name="ResNets50")

    return model

if __name__ == "__main__":
    X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = resnets_utils.load_dataset()
    
    X_train = X_train_orig/255
    X_test = X_test_orig/255
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    temp_model = ResNets50(X_train.shape)
    temp_model.compile("adam","binary_crossentropy", metrics=['accuracy'])
    temp_model.fit(X_train, Y_train, epochs=40, batch_size=50)
    preds = temp_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    print ("erroe = " + str(preds[0]))
    print ("accuracy = " + str(preds[1]))

    img_path = 'images/fingers_big/2.jpg'

    my_image = image.load_img(img_path, target_size=(64, 64))
    my_image = image.img_to_array(my_image)

    my_image = np.expand_dims(my_image,axis=0)
    my_image = preprocess_input(my_image)

    print("my_image.shape = " + str(my_image.shape))

    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(my_image))

    my_image = scipy.misc.imread(img_path)