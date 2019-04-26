#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:24:34 2019

@author: user20
"""

from cnn_model import *
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def cross_entropy(y_predict, label):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_predict, labels=label))
    #優化器: 使cross entropy最小化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #預測準確度
    #argmax: 得到Tensor中最大值的index
    correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(label,1))
    #神經網路準確率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return train_step, accuracy




model = mnist_cnn_model()
y = model.run_cnn()


train_step, accuracy = cross_entropy(y, model.label_placeholder)
train_accuracy = 0.0



#initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#train for 1000 times
for i in range(500):
    batch = mnist.train.next_batch(100)
    #跑訓練集，利用optimizer將結果導到較好的地方
    train_step.run(session=sess, feed_dict={model.img_placeholder: batch[0],
                                            model.label_placeholder: batch[1], model.keep_prob: 0.7})
    
    if i%10 == 0:
        #輸出準確度
        train_accuracy = accuracy.eval(session=sess, feed_dict={model.img_placeholder:batch[0], 
                                        model.label_placeholder: batch[1], model.keep_prob: 1.0})
        
        print("step {}, training accuracy {:}".format(i, train_accuracy))

print("Accuracy: ", sess.run(accuracy, feed_dict={model.img_placeholder: mnist.test.images,
                                model.label_placeholder: mnist.test.labels, model.keep_prob: 1.0}))





print("test:", sess.run(tf.argmax(mnist.test.labels,1)))
print("model:", sess.run(tf.argmax(y, 1), feed_dict={model.img_placeholder: mnist.test.images,
                                            model.label_placeholder: mnist.test.labels, 
                                            model.keep_prob: 1.0}))


# Saving

export_dir =  "my_net/save_net.ckpt"
# saver = tf.train.Saver()
saver = tf.train.Saver({
            "W_conv1": model.W_conv1, 
            "W_conv2": model.W_conv2, 
            "b_conv1": model.b_conv1, 
            "b_conv2": model.b_conv2, 
            "W_fc1": model.W_fc1, 
            "W_fc2": model.W_fc2, 
            "b_fc1": model.b_fc1, 
            "b_fc2": model.b_fc2
        })
sess.run(init)
save_path = saver.save(sess, export_dir)
#save_path = saver.save(sess, export_dir, global_step = 200)
print("Model saved in path: ", save_path)


