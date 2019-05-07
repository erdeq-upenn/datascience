# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:56:47 2019

@author: Dequan Er
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

x = np.random.rand(100)
y = np.random.rand(100)

#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#
#def animate(i):
#    ax1.clear()
#    for i in range(100):
#        ax1.scatter(x[i],y[i])
#
#ani = animation.FuncAnimation(fig,animate,interval=100)
#plt.show()

fig,ax = plt.subplots()

x=np.arange(0,2*np.pi,0.001)
line,=ax.plot(x,np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/100))
    return line,


def init():
    line.set_ydata(np.sin(x))
    return line,
    
    
ani = animation.FuncAnimation(fig=fig,func=animate,frames=100,init_func=init,
                              interval=20,blit=False )
plt.show()
HTML(ani.to_html5_video())