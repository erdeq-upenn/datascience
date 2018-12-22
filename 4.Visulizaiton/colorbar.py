import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

x = np.linspace(0, 10, 1000)
# print x.shape
I = np.sin(x) * np.cos(x[:, np.newaxis])
# print I.shape
# plt.imshow(I)
# plt.colorbar()
#*************
# make noise in 10% of the image pixels
# speckles = (np.random.random(I.shape) < 0.1)
# I[speckles] = np.random.normal(0, 0.5, np.count_nonzero(speckles))
#
# plt.figure(figsize=(10, 3.5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar()
#
# plt.subplot(1, 2, 2)
# plt.imshow(I, cmap='RdBu')
# plt.colorbar(extend='both')
# plt.clim(-1, 1);

# plt.show()
#*****************
# load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])
#***************************
plt.figure()
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
# plot the results
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
#***************************
plt.show()
