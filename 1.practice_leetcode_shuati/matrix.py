import numpy as np
import math


a0 = [ [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
        ]
N=4
for i in range(N):
    for j in range(N):
        print  a0[i][j],
    print
ma0 = np.matrix(a0)
mb0 = [[1,2,3,4]]
print(np.dot(mb0,ma0))
