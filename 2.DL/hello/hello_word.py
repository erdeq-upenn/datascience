# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:36:07 2019

@author: Dequan Er
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
import matplotlib.pyplot as plt

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3


### creat tensorflow sturcture start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
Bias = tf.Variable(tf.zeros([1]))
y = Weights*x_data+Bias

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate 
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() # this is new while tf.initialize_all_variables are depreciated 

### creat tensorflow sturcture end ###

sess = tf.Session()
sess.run(init)  # inital thing of initialization of variables

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(Bias))

# the expected weight and bias are expected to be 0.1 and 0.3
        
matrix1 = tf.constant([[3,3]])   # matrix1.shape (1,2)
matrix2 = tf.constant([[2],      # matrix2.shape (2,1)
                     [2]])
product = tf.matmul(matrix1,matrix2)  # matrix multiplication; in np is np.dot(m1,m2)

# method 1 
sess = tf.Session()
res = sess.run(product)
print(res)
sess.close()

# method 2
with tf.Session() as sess:
    res2 = sess.run(product)
    print(res2)


state =tf.Variable(0,name='counter')
print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state,new_value)

# initialization 

init = tf.global_variables_initializer()  # init
 
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# fit position first and then wait for outside inputs 

input1 = tf.placeholder(tf.float32,)  # assign data type tf.placeholder(tf.float32,[2,2]) two rows and tow columns 
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))


def add_layer(inputs,in_size,out_size,activation_function=None):
    # new
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('bias'):  
            bias = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('w_plus_b'): 
            Wx_plus_bias = tf.add(tf.matmul(inputs,Weights),bias)
        if activation_function is None:
            output = Wx_plus_bias
        else:
            output = activation_function(Wx_plus_bias)
        return output

 # add layer function   
 
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

# new def inpupt networks 
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input') 
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
    
# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)
# with tf.name_scope('loss'): 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# with tf.name_scope('train'): 
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
init =tf.initialize_all_variables()
sess = tf.Session()

# write = tf.train.SummaryWriter('../input/',sess.graph)

sess.run(init)
#for i in range(10000):
#    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#    if i%50==0:
#        print(sess.run(loss,feed_dict={xs: x_data,ys:y_data}))
#         prediction_value = sess.run(prediction,feed_dict={xs:x_data})
#         lines = ax.plot(x_data,prediction_value.'r-',lw=5)


# plotting 
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
# real data

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
#         print(sess.run(loss,feed_dict={xs: x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)




