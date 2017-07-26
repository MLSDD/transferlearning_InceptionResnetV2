#coding=utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import PIL as pillow
from PIL import Image
import numpy as np
import types
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from inception_resnet_v2 import *


with open('imagenet1000_clsid_to_human.txt','r') as inf:
    imagenet_classes = eval(inf.read())

def get_human_readable(id):
    id = id - 1
    label = imagenet_classes[id]
    return label



# inception_resnet_v2
def load_model_inceptionResnetV2():
    checkpoint_file = '/home/cr/PycharmProjects/pre-trainedModel/inception_resnet_v2_2016_08_30.ckpt'
    # loading the inception graph
    arg_scope = inception_resnet_v2_arg_scope()
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(inputs=input_tensor, is_training=False, num_classes=1001)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, checkpoint_file)
    return sess, logits, end_points, input_tensor

def classifyImages_inceptionResnetV2(sess, input_tensor, logits, end_points, sample_images):
    classifications = []
    for image in sample_images:
            im = Image.open(image).resize((299,299))
            im = np.array(im)
            im = im.reshape(-1,299,299,3)
            im = 2*(im/255.0)-1.0
            predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})

            '''
              print(predict_values)
              print (np.max(predict_values), np.max(logit_values))
              print ((np.argmax(predict_values) - 1), (np.argmax(logit_values) - 1))          # np.argmax(predict_values)�?开始，而实际内容从0开始，因此要减�?
              '''
            # 输入最大的前N个概率对应的类别
            N = 10
            predict_values = predict_values[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-predict_values), key=lambda x: x[1])]
            for i in range(N):
                index = sorted_inds[i]
                print((predict_values[index], imagenet_classes[index - 1]))


            # 输入最大概率对应的类别
            label = get_human_readable(np.argmax(predict_values))
            predict_value = np.max(predict_values)
            classifications.append({"label": label, "predict_value": predict_value})
            print (classifications)




if __name__ == "__main__":
    sample_image = ['/home/cr/Downloads/bird.jpg']
    sess, logits, end_points, input_tensor = load_model_inceptionResnetV2()
    classifyImages_inceptionResnetV2(sess, input_tensor, logits, end_points, sample_image)



