import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 1000)
# fig, ax = plt.subplots()
# ax.plot(x, np.sin(x), '-b', label='Sin')
# ax.plot(x, np.cos(x), '--r', label='Cos')
# ax.axis('equal')
# leg = ax.legend();
# ax.legend(loc='upper left', frameon=False)
# fig
# ax.legend(frameon=False, loc='lower center', ncol=2)
# fig
# ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
# fig
# y = np.sin(x[:,np.newaxis]+np.pi *np.arange(0,2,0.5))
# lines = plt.plot(x,y)
# plt.legend(lines[:2],['first','second'])
# plt.axes([0, 10, -1, 1])
####################
fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
colors = ['r','g','b','c']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color = colors[i])
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
          loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
# leg = Legend(ax, lines[2:], ['line C', 'line D'],
#              loc='lower right', frameon=False)
# ax.add_artist(leg);
plt.show()
