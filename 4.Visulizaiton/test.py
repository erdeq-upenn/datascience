import numpy as np

# a = np.array([[1,2,3,4],[5,7,8,6]])
# b = np.array([1, 2, 3, 4], dtype='float32')
# c = np.array([range(i,i+4) for i in [3,6,9]])
# print a, b, '\n', c
# print np.zeros(10, dtype=int)
# print np.arange(0,20,2)
# print np.linspace(0, 1, 5)
# x = np.random.random((3, 3))
# y = x**2
# print np.dot(x, y)
# print '$'*10
# print np.random.normal(0, 1, (3, 3))
# np.random.normal(0, 1, (3, 3))
# print np.eye(3)
# print '$'*40
np.random.seed(0)

x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
# print x1, x2, x3
# print "x3 ndim: ", x3.ndim, "x2 ndim: ", x2.ndim, "x1 ndim: ", x1.ndim
# print "x3 shape:", x3.shape
# print "x3 size: ", x3.size

# print "itemsize:", x3.itemsize, "bytes"
# print "nbytes:", x3.nbytes, "bytes"
# print x1, x1[0], x1[-1], x1[::-1]
#
# print x2
# print x2[::-1,::-1]
# print x2[:,0] #print fisrt coloum

# Subarrays as no-copy views

# Creating copies of arrays

# x2_sub_copy = x2[:2,:2].copy()
# x2_sub_copy[0,0] = 999
# print x2_sub_copy
# print x2
x = np.array([1,2,3])
grid = np.array([[9,8,7],[2,5,1]])
# print grid
#print np.vstack([grid,x])
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1,x2 = np.split(x,[4])
#print(x1,x2)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig =plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
