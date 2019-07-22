"""RBF interpolation"""

from .interpolate import BaseInterpolation
import numpy as np


_MOD_RBF_WEIGHT = """
/**
 * Weight function kernel
 * x1 : first point lat
 * y1 : first point long
 * x2 : second point lat
 * y2 : second point long
 * epsilon : RBF shape parameter
 */
__device__ {f} weight_function({f} x1, {f} y1, {f} x2, {f} y2, {f} epsilon)
{{
    return exp(-1  * epsilon * (pow(x1 - x2, 2) + pow(y1 - y2, 2)));
}}
"""


class RBFInterpolation(BaseInterpolation):
    WEIGHT_FUNCTION = _MOD_RBF_WEIGHT

    OTHER_ARGS = "{f} epsilon"
    OTHER_WF_ARGS = "epsilon"

    def get_other_args(self, epsilon=1.0):
        return [self.FLOAT_TYPE(epsilon)]

    def weights(self, latitudes, longitudes, values, epsilon=1.0):
        """Get RBF weights w_i:

        [RBF(||x_i - x_j||)]_i,j [w_i].T = [f(x_i)]
        """

        n = len(values)
        mat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dist = (
                    (latitudes[i] - latitudes[j])**2 +
                    (longitudes[i] - longitudes[j])**2
                )**0.5
                mat[i, j] = np.exp(-1 * epsilon * dist)

        return np.matmul(np.linalg.inv(mat), values)
