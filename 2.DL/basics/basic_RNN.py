# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:01:02 2019

@author: Dequan Er
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2

#X0 = tf.placeholder(tf.float32,[None,n_inputs])
#X1 = tf.placeholder(tf.float32,[None,n_inputs])
#
#
#Wx = tf.Variable(tf. _normal(shape = [n_inputs,n_neurons],dtype=tf.float32))
#Wy = tf.Variable(tf.random_normal(shape = [n_neurons,n_neurons],dtype=tf.float32))
#b  = tf.Variable(tf.zeros([1,n_neurons],dtype=tf.float32))
##
#Y0 = tf.tanh(tf.matmul(X0,Wx)+b)
#Y1 = tf.tanh(tf.matmul(Y0,Wy)+b)
#
#init = tf.global_variables_initializer()
#
## mini-batch 
#
#X0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]]) # t = 0 instance
#X1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]]) # t = 1 instance
#
#with tf.Session() as sess:
#    init.run()
#    Y0_val,Y1_val = sess.run([Y0,Y1],feed_dict = {X0:X0_batch,X1:X1_batch})
#    
#print(Y0_val.shape,Y1_val.shape)

####################stage I#########################################
# use static_rnn() creates the same unrolled RNN 
#X0 = tf.placeholder(tf.float32,[None,n_inputs])
#X1 = tf.placeholder(tf.float32,[None,n_inputs])
#
##
##
#basic_cell = tf.keras.layers.SimpleRNNCell(n_neurons)
#output_seqs, states = tf.nn.static_rnn(basic_cell,[X0,X1],dtype=tf.float32)
#Y0,Y1 = output_seqs
#
#
#
#X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
#X_seqs = tf.unstack(tf.transpose(X,perm = [1,0,2]))
#basic_cell = tf.keras.layers.SimpleRNNCell(n_neurons)
#outputs = tf.transpose(tf.stack(output_seqs),perm = [1,0,2])
#
#X_batch = np.array([
#        # t =0    t = 1
#        [[0,1,2],[9,8,7]],
#        [[3,4,5],[0,0,0]],
#        [[6,7,8],[6,5,4]],
#        [[9,0,1],[3,2,1]],
#        ])
#
#with tf.Session() as sess:
#    init.run()
#    outputs_val = outputs.eval(feed_dict={X:X_batch})
#    
#    
    
#############################################################
    
# MNIST sequence classifier

#n_steps = 28
#n_inputs =28 
#n_neurons = 150
#n_outputs = 10
#learning_rate = 0.001
#
#X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
#y = tf.placeholder(tf.int32,[None])
#
#basic_cell = tf.keras.layers.SimpleRNNCell(n_neurons)
#outputs, states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)
#
#logits = tf.layers.dense(states,n_outputs)
#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
#
#loss = tf.reduce_mean(xentropy)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
#training_op = optimizer.minimize(loss)
#
#correct = tf.nn.in_top_k(logits,y,1)
#accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
#
#init = tf.global_variables_initializer()
#
#
#
#
##input data  
#from tensorflow.examples.tutorials.mnist import input_data
#
##mnist = input_data.read_data_sets('\data')
#
#X_test = mnist.test.images.reshape((-1,n_steps,n_inputs))
#y_test = mnist.test.labels
#
#g = plt.imshow(X_test[0])
################################################################################
#training

n_epochs = 100
batch_size = 150

# run RNN on tensorflow
#with tf.Session() as sess:
#    init.run()
#    for epoch in range(n_epochs):
#        for iteration in range(mnist.train.num_examples//batch_size):
#            X_batch,y_batch = mnist.train.next_batch(batch_size)
#            X_batch = X_batch.reshape((-1,n_steps,n_inputs))
#            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
#        acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
#        acc_test = accuracy.eval(feed_dict={X:X_test,y:y_test})
#        print(epoch,'Train accuracy:',acc_train,'test accuracy:',acc_test)


# RNN training to predict time series

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1) 

t = np.linspace(t_min, t_max, (t_max - t_min) // resolution)

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")


#save_fig("time_series_plot")
plt.show()


X_batch, y_batch = next_batch(1, n_steps)

#Using an OuputProjectionWrapper


tf.reset_default_graph()

from tensorflow.contrib.layers import fully_connected

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

n_outputs = 1
learning_rate = 0.001

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 1000
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred)
# plotting 
plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

#save_fig("time_series_pred_plot")
plt.show()


#Without using an OutputProjectionWrapper
tf.reset_default_graph()

from tensorflow.contrib.layers import fully_connected

n_steps = 20
n_inputs = 1
n_neurons = 100

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

n_outputs = 1
learning_rate = 0.001

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn=None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


n_iterations = 1000
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred)

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()



#Generating a creative new sequence

n_iterations = 2000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "r-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "r-", linewidth=3)
plt.xlabel("Time")
#save_fig("creative_sequence_plot")
plt.show()