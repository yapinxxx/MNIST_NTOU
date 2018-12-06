
# coding: utf-8

# In[9]:


#creating cnn model

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# weight_variable: -0.2~0.2
#將shape的每個element設為從常態分佈隨機取一個數, 大於2個標準差的數會drop, re-pick
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#bias = 0.1
#Creates a constant tensor and initialize it to 0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#2-D CNN
#stride(位移量) = [1, stride,stride, 1]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

#x(input) = [batch, height, width, channels]
#ksize(窗口大小) = [1, height, width, 1]
#stride = [1, stride,stride, 1]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')





def cnn_pooling(W, b, x_image):
    h_conv = tf.nn.relu(conv2d(x_image, W) + b)
    h_pool = max_pool_2x2(h_conv)
    return h_pool


def full_connect(W_fc, b_fc, h_pool):
    h_pool_flat = tf.reshape(h_pool, [-1, 7 * 7 * 64])
    #NOT CNN!!
    h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc)
    return h_fc

def drop_out(h_fc, keep_prob):
    #dropout
    #避免過度擬合(over fitting)
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    return h_fc_drop

def softmax(W_fc, b_fc, h_fc_drop):
    #softmax
    #轉為0~1的機率
    y_predict = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc)+b_fc)
    return y_predict



#the whole cnn model
class mnist_cnn_model:
    
    def __init__(self):
        
        self.img_placeholder = tf.placeholder(tf.float32, [None, 784])
        self.label_placeholder = tf.placeholder(tf.float32, [None, 10])
        #input dimantion: -1(不考慮維度), size: 28*28, channel:1
        self.input_img = tf.reshape(self.img_placeholder, [-1, 28, 28, 1])
        
        self.W_conv1 = weight_variable([5, 5, 1, 32]) 
        self.b_conv1 = bias_variable([32])
        self.W_conv2 = weight_variable([5, 5, 32, 64]) 
        self.b_conv2 = bias_variable([64])
    
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])
        
        self.keep_prob =  tf.placeholder(tf.float32)

    def set_value(self, W_conv1, W_conv2, b_conv1, b_conv2, W_fc1, W_fc2, b_fc1, b_fc2):
        self.W_conv1 = W_conv1
        self.b_conv1 = b_conv1
        self.W_conv2 = W_conv2
        self.b_conv2 = b_conv2
        self.W_fc1 = W_fc1
        self.b_fc1 = b_fc1
        self.W_fc2 = W_fc2
        self.b_fc2 = b_fc2
        return
    
    def run_cnn(self):
        h_pool1 = cnn_pooling(self.W_conv1, self.b_conv1, self.input_img)
        h_pool2 = cnn_pooling(self.W_conv2, self.b_conv2, h_pool1)
        
        h_fc1 = full_connect(self.W_fc1, self.b_fc1, h_pool2)
        h_fc_drop1 = drop_out(h_fc1, self.keep_prob)
        
        y_predict = softmax(self.W_fc2, self.b_fc2, h_fc_drop1)
     
        return y_predict
    
    def details(self):
        print("{:30}:{}".format("convolution weight 1", self.W_conv1))
        print("{:30}:{}".format("convolution bias 1", self.b_conv1))
        print("{:30}:{}".format("convolution weight 2", self.W_conv2))
        print("{:30}:{}".format("convolution bias 2", self.b_conv2))
        
        print("{:30}:{}".format("full connected weight 1", self.W_fc1))
        print("{:30}:{}".format("full connected bias 1", self.b_fc1))
        print("{:30}:{}".format("full connected weight 2", self.W_fc2))
        print("{:30}:{}".format("full connected bias 2", self.b_fc2))
        return
        

