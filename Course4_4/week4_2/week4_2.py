import time
import os
import sys
from colorama import Style

from matplotlib import image
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import nst_utils
import numpy as np
import tensorflow as tf
from week4_2.nst_utils import generate_noise_image, reshape_and_normalize_image

def compute_content_cost(a_C,a_G):
    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    #对a_C与a_G从3维降到2维
    a_C_unrolled = tf.transpose(tf.reshape(a_C,[n_H*n_W,n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,[n_H*n_W,n_C]))

    #计算内容代价
    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))

    return J_content

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    
    return GA

def compute_style_cost(model,STYLE_LAYERS):
    J_style = 0
    for layer_name,coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S,a_G)
        J_style += coeff*J_style_layer
    
    return J_style

def total_cost(J_content,J_style,alpha=10,beta=40):
    J = alpha*J_content+beta*J_style

    return J

if __name__ == "__main__":
    STYLE_LAYERS = [
        ('conv1_1',0.2),
        ('conv2_1',0.2),
        ('conv3_1',0.2),
        ('conv4_1',0.2),
        ('conv5_1',0.2)
    ]

    content_image = scipy.misc.imread("D:/deep learning course/Course4/week4_2/images/louvre.jpg")
    content_image = reshape_and_normalize_image(content_image)
    style_image = scipy.misc.imread("D:/deep learning course/Course4/week4_2/images/monet_800600.jpg")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    model = nst_utils.load_vgg_model("D:/deep learning course/Course4/week4_2/pretrained-model/imagenet-vgg-verydeep-19.mat")

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C,a_G)
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model,STYLE_LAYERS)

    J = total_cost(J_content,J_style,10,40)
    