import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('classic')
import numpy as np


# x = np.random.randn(1000)
#
# # use a gray background
# ax = plt.axes()
# ax.set_axisbelow(True)
#
# # draw solid white grid lines
# plt.grid(color='w', linestyle='solid')
#
# # hide axis spines
# for spine in ax.spines.values():
#     spine.set_visible(False)
#
# # hide top and right ticks
# ax.xaxis.tick_bottom()
# ax.yaxis.tick_left()
#
# # lighten ticks and labels
# ax.tick_params(colors='gray', direction='out')
# for tick in ax.get_xticklabels():
#     tick.set_color('gray')
# for tick in ax.get_yticklabels():
#     tick.set_color('gray')
#
# # control face and edge color of histogram
# ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');
#
###################
fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis');

fig2 =plt.figure()
x = [0.5,0.2,0.1,0.1,0.05,0.05]
plt.pie(x)

###################
plt.show()
