import matplotlib.pyplot as plt
import numpy as np
import schrodinger as sc
from math import pi

domain = sc.Rectangle(0, pi, 0, pi)

s = sc.Schrodinger2D(lambda x, y: 0, domain, gridSize=(30, 30))

efs = sorted(s.eigenfunctions(), key=lambda x: x[0])[:10]

e, f = efs[0]

x = np.linspace(0, 0.4, 100)
y = np.linspace(0, 0.4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X.flatten(), Y.flatten()).reshape(X.shape)[::-1, :]

plt.imshow(Z, cmap="hsv", extent=(np.min(x), np.max(x), np.min(y), np.max(y)))
plt.show()
