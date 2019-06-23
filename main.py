import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from matplotlib import pyplot as plt
import numpy as np


_MOD_IDW_WEIGHT = """
__device__ {f} weight_function({f} x1, {f} y1, {f} x2, {f} y2) {{

    {f} sum = pow(x1 - x2, 2) + pow(y1 - y2, 2);

    if(sum < 0.000001) {{ return 1000000; }}
    else {{ return rsqrt(sum); }}
}}
"""

_MOD_INTERP = """
__global__ void interpolate(
        {f} *lat_arr, {f} *long_arr, {f} *val, int size,
        {f} *res, int width_px, int height_px,
        {f} lat_min, {f} long_min, {f} lat_max, {f} long_max)
{{
    int px_x = threadIdx.x + blockIdx.x * blockDim.x;
    int px_y = threadIdx.y + blockIdx.y * blockDim.y;

    if((px_x < height_px) && (px_y < width_px)) {{

        {f} lat_pt = lat_max - (lat_max - lat_min) / height_px * px_x;
        {f} long_pt = long_min + (long_max - long_min) / width_px * px_y;

        {f} acc;
        {f} total_weight;
        {f} weight;

        for(int i = 0; i < size; i++) {{
            weight = weight_function(lat_arr[i], long_arr[i], lat_pt, long_pt);
            total_weight += weight;
            acc += weight * val[i];
        }}

        res[px_x * width_px + px_y] = acc / total_weight;
    }}
}}
"""


class KernelConfigurationException(Exception):
    """Exception raised by a misconfigured GPU kernel"""
    pass


class IDWInterpolation:

    FLOAT_SIZES = {
        32: [np.float32, "float"],
        64: [np.float64, "double"]
    }

    def __init__(self, fsize=32, block=(16, 16, 1)):

        if block[2] != 1:
            raise KernelConfigurationException(
                "Block size dimension 3 must be 1.")

        if (block[0] * block[1]) % 32 != 0:
            raise KernelConfigurationException(
                "Kernel size must be divisible by 32.")

        if fsize not in self.FLOAT_SIZES:
            raise KernelConfigurationException(
                "Float size must be 32 or 64.")

        self.FLOAT_TYPE, self.FLOAT_CTYPE = self.FLOAT_SIZES[fsize]
        self.BLOCK_SIZE = block

        self.__kernel = SourceModule(
            (_MOD_IDW_WEIGHT + _MOD_INTERP).format(f=self.FLOAT_CTYPE)
        ).get_function("interpolate")

    def __get_grid(self, size):
        """Get grid dimensions for a GPU operation.

        Parameters
        ----------
        size : int[2]
            Target image size

        Returns
        -------
        int[3]
            PyCuda BlockSize argument
        """
        return (
            int(np.ceil(size[0] / self.BLOCK_SIZE[0])),
            int(np.ceil(size[1] / self.BLOCK_SIZE[1])),
            1)

    def interpolate(
            self, latitudes, longitudes, values,
            size=(1024, 1024), lat_range=(0, 1), long_range=(0, 1)):
        """Run interpolation

        Parameters
        ----------
        latitudes : np.array
            Latitude vector; array with dtype float32 or float64
        longitudes : np.array
            Longitude vector
        values : np.array
            Point values
        size : int[2]
            Output image size
        lat_range : float[2]
            Range of latitudes to capture. lat_range[0] corresponds to
            result[height], and lat_range[1] corresponds to result[0].
        long_range : float[2]
            Range of longitudes to capture. long_range[0] corresponds to
            result[:, 0], and long_range[1] corresponds to result[:, width].

        Returns
        -------
        np.array
            Generated rasterized interpolation image.
        """

        # Check shapes
        if(
                (latitudes.shape[0] != longitudes.shape[0]) or
                (values.shape[0] != latitudes.shape[0])):
            raise TypeError("Coordinate dimension mismatch")

        # Check types
        if(
                (latitudes.dtype != self.FLOAT_TYPE) or
                (longitudes.dtype != self.FLOAT_TYPE) or
                (values.dtype != self.FLOAT_TYPE)):
            raise TypeError(
                "Input data must be of type {}. To change the input type, set"
                "the 'fsize' argument on initialization.".format(
                    self.FLOAT_TYPE.__name__))

        # Allocate and transfer data to GPU
        lat_gpu = cuda.mem_alloc(latitudes.nbytes)
        long_gpu = cuda.mem_alloc(longitudes.nbytes)
        val_gpu = cuda.mem_alloc(values.nbytes)
        cuda.memcpy_htod(lat_gpu, latitudes)
        cuda.memcpy_htod(long_gpu, longitudes)
        cuda.memcpy_htod(val_gpu, values)

        # Set up result
        res_cpu = np.zeros(size, dtype=self.FLOAT_TYPE)
        res_gpu = cuda.mem_alloc(res_cpu.nbytes)

        # Run kernel
        self.__kernel(
            lat_gpu, long_gpu, val_gpu, np.int32(latitudes.shape[0]),
            res_gpu, np.int32(res_cpu.shape[1]), np.int32(res_cpu.shape[0]),
            self.FLOAT_TYPE(lat_range[0]), self.FLOAT_TYPE(long_range[0]),
            self.FLOAT_TYPE(lat_range[1]), self.FLOAT_TYPE(long_range[1]),
            block=self.BLOCK_SIZE, grid=self.__get_grid(size))

        # Fetch result
        cuda.memcpy_dtoh(res_cpu, res_gpu)

        return res_cpu


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
    # plt.scatter((coords[1] + 2) / 11 * 256, (9 - (coords[0] + 3)) / 9 * 128)
    plt.show()
