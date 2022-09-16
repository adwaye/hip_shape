import numpy as np
from mayavi import mlab
x, y, z, value = np.random.random((4, 40))
mlab.points3d(x, y, z, value)