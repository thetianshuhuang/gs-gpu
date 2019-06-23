
import numpy as np
from matplotlib import pyplot as plt

from gs_gpu import IDWInterpolation

if __name__ == '__main__':

    coords = [
        np.array([0, 0, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    ]

    values = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)

    res = IDWInterpolation(fsize=64).interpolate(
        coords[0], coords[1], values,
        size=[128, 256], lat_range=[-3, 6], long_range=[-2, 9])

    plt.imshow(res)
    plt.scatter((coords[1] + 2) / 11 * 256, (9 - (coords[0] + 3)) / 9 * 128)
    plt.show()
