import os
import numpy as np
from keras import backend as K
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model
import argparse
import os
import scipy.io
import scipy.misc
import pandas as pd
import tensorflow as tf

from yad2k.models.keras_yolo import yolo_head,yolo_boxes_to_corners,preprocess_true_boxes,yolo_loss,yolo_body

import yolo_utils

def yolo_filter_boxes(box_confidence,boxes,box_class_probs,threshold=0.6):
    #第一步：计算锚框的得分
    box_scores = box_confidence*box_class_probs

    #第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)

    #第三步：根据阈值创建掩码
    filtering_mask = (box_class_scores>=threshold)

    #对scores, boxes 以及 classes使用掩码
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)

    return scores,boxes,classes

def iou(box1,box2):
    #计算相交的区域的面积
    xi1 = np.maximum(box1[0],box2[0])
    yi1 = np.maximum(box1[1],box2[1])
    xi2 = np.minimum(box1[2],box2[2])
    yi2 = np.minimum(box1[3],box2[3])
    inter_area = (xi1-xi2)*(yi1-yi2)

    #计算并集，公式为：Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2]-box2[0])*(box1[3]-box2[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area+box2_area-inter_area

    #计算交并比
    iou = inter_area/union_area

    return iou

def yolo_non_max_suppression(scores,boxes,classes,max_boxes=10,iou_threshold=0.5):
    #为锚框实现非最大值抑制
    max_boxes_tensor = K.variable(max_boxes,dtype="int32")        #实例化一个张量
    K.get_session().run(tf.variable_initializer([max_boxes_tensor]))

    #使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,iou_threshold)

    #使用K.gather()来选择保留的锚框
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)

    return scores,boxes,classes

def yolo_eval(yolo_outputs,image_shape=(720.,1280.),max_boxes=10,score_threshold=0.6,iou_threshold=0.5):
    #将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

    #获取YOLO模型的输出
    box_confidence,box_xy,box_wh,box_class_probs = yolo_outputs

    #中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy,box_wh)

    #可信度分值过滤
    scores,boxes,classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)

    #缩放锚框，以适应原始图像
    boxes = yolo_utils.scale_boxes(boxes,image_shape)

    #使用非最大值抑制
    scores,boxes,classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)

    return scores,boxes,classes

def predict(sess,image_file,is_show_info=True,is_plot=True):
    #图像预处理
    image,image_data = yolo_utils.preprocess_image("images/"+image_file,model_image_size=(608,608))

    #运行会话并在feed_dict中选择正确的占位符
    out_scores,out_boxes,out_classes = sess.run([scores,boxes,classes],feed_dict={yolo_model.input:image_data,K.learning_phase():0})

    #打印预测信息
    if is_show_info:
        print(str(image_file)+"find "+str(len(out_boxes))+" boxes")
    
    # 指定要绘制的边界框的颜色
    colors = yolo_utils.generate_colors(class_names)

    #在图中绘制边界框
    yolo_utils.draw_boxes(image,out_scores,out_boxes,out_classes,class_names,colors)

    #保存已经绘制了边界框的图`
    image.save(os.path.join("out",image_file),quality=100)

    #打印出已经绘制了边界框的图
    if is_plot:
        output_image = scipy.misc.imread(os.join("out",image_file))
    
    return out_scores,out_boxes,out_classes


if __name__ == "__main__":
    sess = K.get_session()
    class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
    anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720.,1280.)
    yolo_model = load_model("model_data/yolov2.h5")
    yolo_model.summary()

    yolo_outputs = yolo_head(yolo_model.output,anchors,len(class_names))

    scores,boxes,classes = yolo_eval(yolo_outputs,image_shape)