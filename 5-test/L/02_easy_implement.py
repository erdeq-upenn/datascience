#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:00:26 2020

@author: Dequan
Simple implementation of NN
"""

from mxnet import autograd, nd
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

#
## generating features
#num_inputs = 2
#num_examples = 1000
#true_w = [2, -3.4]
#true_b = 4.2
#features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
#labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
#labels += nd.random.normal(scale=0.01, shape=labels.shape)
#
## read data 
#
#from mxnet.gluon import data as gdata
#
#batch_size = 10
## 将训练数据的特征和标签组合
#dataset = gdata.ArrayDataset(features, labels)
## 随机读取小批量
#data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
#
#for X, y in data_iter:
#    print(X, y)
#    break
#
## model define
#net = nn.Sequential()
#net.add(nn.Dense(1))
#net.initialize(init.Normal(sigma=0.01))
#loss = gloss.L2Loss()  # 平方损失又称L2范数损失
#trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
#
## model training 
#num_epochs = 3
#for epoch in range(1, num_epochs + 1):
#    for X, y in data_iter:
#        with autograd.record():
#            l = loss(net(X), y)
#        l.backward()
#        trainer.step(batch_size)
#    l = loss(net(features), labels)
#    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
#
#
#dense = net[0]
#print("True", true_w, "\n", "Trained", dense.weight.data())
#print(true_b, dense.bias.data())
#

########################
# minist fashion 
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

batch_size = 256
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

feature, label = mnist_train[0]
print(feature.shape, feature.dtype)

def get_fashion_mnist_labels(labels):
    
    """define lable and corresponding words"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

# visualize data 
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# read mini-batch
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
# read data time
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec for data loading' % (time.time() - start)

# model init
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

#accuracy(y_hat, y)

# 描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n

evaluate_accuracy(test_iter, net)


num_epochs, lr = 10, 0.01

# training the model
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # “softmax回归的简洁实现”一节将用到
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)

# prediction 
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])



