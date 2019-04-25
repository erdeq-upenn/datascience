# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:28:07 2019

@author: Dequan Er
"""

from functools import partial 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X= tf.placeholder(tf.float32,shape=[None,n_inputs])
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense = partial(tf.layers.dense,
                   activation=tf.nn.relu,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2_regularizer)

hidden1= my_dense(X,n_hidden1)
hidden2= my_dense(hidden1,n_hidden2)
hidden3= my_dense(hidden2,n_hidden3)
outputs= my_dense(hidden3,n_outputs,activation=None)

recon_loss =tf.reduce_mean(tf.square(outputs-X)) ## MSE

reg_losss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([recon_loss]+reg_losss)

optimizer = tf.train.AdadeltaOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()



#input data  
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets('\data')

#X_test = mnist.test.images.reshape((-1,n_steps,n_inputs))
#y_test = mnist.test.labels


n_epochs = 5
batch_size =150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples //batch_size
        for iteration in range(n_batches):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch})
            if iteration%50 == 0:
                mse = loss.eval(feed_dict={X: X_batch})
                print(iteration, "\tMSE:", mse)
                p = sess.run(outputs, feed_dict={X: X_batch}) 