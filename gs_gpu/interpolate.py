"""Common interpolation routines

Attributes
----------
_MOD_INTERP : str
    Base interpolation kernel. Note that this takes the form of a format
    string, and must be formatted with {f}=float or double when used.
KernelConfigurationException : exception
    Exception type raised by kernel wrappers
BaseInterpolation : class
    Base Interpolation class; handles all wrapping (extenders only need to
    provide a weight function).
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np


#
# -- Main Kernel --------------------------------------------------------------
#

_MOD_INTERP = """
/**
 * Interpolation kernel
 * lat_arr : latitude array
 * long_arr : longitude array
 * val : value array
 * weights : weights array; set to all ones if no weighting desired
 * size : number of points
 * res : result array
 * width_px : result raster image width
 * height_px : result raster image height
 * lat_min : latitude value corresponding to image bottom
 * long_min : longitude value corresponding to image left
 * lat_max : latitude value corresponding to image top
 * long_max : longitude value corresponding to image right
 * {other_args} : other parameters
 */
__global__ void interpolate(
        {f} *lat_arr, {f} *long_arr, {f} *val, {f} *weights, int size,
        {f} *res, int width_px, int height_px,
        {f} lat_min, {f} long_min, {f} lat_max, {f} long_max,
        {other_args})
{{
    int px_x = threadIdx.x + blockIdx.x * blockDim.x;
    int px_y = threadIdx.y + blockIdx.y * blockDim.y;

    if((px_x < height_px) && (px_y < width_px)) {{

        {f} lat_pt = lat_max - (lat_max - lat_min) / height_px * px_x;
        {f} long_pt = long_min + (long_max - long_min) / width_px * px_y;

        {f} acc = 0;
        {f} total_weight = 0;
        for(int idx = 0; idx < size; idx++) {{
            {f} weight = weights[idx] * weight_function(
                lat_arr[idx], long_arr[idx], lat_pt, long_pt,
                {other_args_as_param});
            total_weight += weight;
            acc += weight * val[idx];
        }}

        res[px_x * width_px + px_y] = acc / total_weight;
    }}
}}
"""


#
# -- Python Wrapper -----------------------------------------------------------
#

class KernelConfigurationException(Exception):
    """Exception raised by a misconfigured GPU kernel"""
    pass


class BaseInterpolation:
    """Base Interpolation class

    Parameters
    ----------
    fsize : int
        32 or 64; size of floating point type
    block : int[3]
        Block size. Should be a multiple of 32 (as per NVIDIA warp size
        specifications). Defaults to 16*16 since 256 is around the recommended
        size (128-512, with 512 being the limit for some cards)
    """

    FLOAT_SIZES = {
        32: [np.float32, "float"],
        64: [np.float64, "double"]
    }

    WEIGHT_FUNCTION = None

    def __init__(self, fsize=32, block=(16, 16, 1)):

        # Check validity
        if block[2] != 1:
            raise KernelConfigurationException(
                "Block size dimension 3 must be 1.")

        if (block[0] * block[1]) % 32 != 0:
            raise KernelConfigurationException(
                "Kernel size must be divisible by 32.")

        if fsize not in self.FLOAT_SIZES:
            raise KernelConfigurationException(
                "Float size must be 32 or 64.")

        # Set parameters
        self.FLOAT_TYPE, self.FLOAT_CTYPE = self.FLOAT_SIZES[fsize]
        self.BLOCK_SIZE = block

        # Get kernel
        self.__init_kernel()

    def __init_kernel(self):
        """Initialize kernel.

        References the WEIGHT_FUNCTION attribute, which should be set by all
        classes extending BaseInterpolation.
        """

        if self.WEIGHT_FUNCTION is None:
            raise KernelConfigurationException(
                "Attempted to initialize kernel with undefined weight "
                "function. If you are writing your own weight_function, make "
                "sure to set the WEIGHT_FUNCTION attribute.")

        # other_args : {type} param, {type} param, ...
        # other_args_as_param : param, param, ...
        kernel_src = (
            self.WEIGHT_FUNCTION.format(
                f=self.FLOAT_CTYPE) +
            _MOD_INTERP.format(
                other_args=self.OTHER_ARGS.format(f=self.FLOAT_CTYPE),
                other_args_as_param=self.OTHER_WF_ARGS,
                f=self.FLOAT_CTYPE)
        )

        self.__kernel = SourceModule(kernel_src).get_function("interpolate")

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
            self, latitudes, longitudes, values, weights=None,
            size=(1024, 1024), lat_range=(0, 1), long_range=(0, 1), **kwargs):
        """Run interpolation

        Parameters
        ----------
        latitudes : np.array
            Latitude vector; array with dtype float32 or float64
        longitudes : np.array
            Longitude vector
        values : np.array
            Point values
        weights : np.array or None
            Point weights; if None, uses uniform weighting
        size : int[2]
            Output image size
        lat_range : float[2]
            Range of latitudes to capture. lat_range[0] corresponds to
            result[height], and lat_range[1] corresponds to result[0].
        long_range : float[2]
            Range of longitudes to capture. long_range[0] corresponds to
            result[:, 0], and long_range[1] corresponds to result[:, width].

        Raises
        ------
        TypeError
            Dimensions don't match, or if the data type does not match the
            data type specified on module initialization.

        Returns
        -------
        np.array
            Generated rasterized interpolation image.
        """

        # Weights
        if weights is None:
            # Kernel-specific weight function
            if hasattr(self, 'weights'):
                weights = self.weights(latitudes, longitudes, values, **kwargs)
            # All weighted equally
            else:
                weights = np.ones(values.shape, dtype=self.FLOAT_TYPE)

        # Check shapes
        if(
                (latitudes.shape[0] != longitudes.shape[0]) or
                (values.shape[0] != latitudes.shape[0]) or
                (weights.shape[0] != values.shape[0])):
            raise TypeError("Coordinate dimension mismatch")

        # Check types
        if(
                (latitudes.dtype != self.FLOAT_TYPE) or
                (longitudes.dtype != self.FLOAT_TYPE) or
                (values.dtype != self.FLOAT_TYPE) or
                (weights.dtype != self.FLOAT_TYPE)):
            raise TypeError(
                "Input data must be of type {}. To change the input type, set "
                "the 'fsize' argument on initialization.".format(
                    self.FLOAT_TYPE.__name__))

        # Allocate and transfer data to GPU
        lat_gpu = cuda.mem_alloc(latitudes.nbytes)
        long_gpu = cuda.mem_alloc(longitudes.nbytes)
        val_gpu = cuda.mem_alloc(values.nbytes)
        weights_gpu = cuda.mem_alloc(weights.nbytes)
        cuda.memcpy_htod(lat_gpu, latitudes)
        cuda.memcpy_htod(long_gpu, longitudes)
        cuda.memcpy_htod(val_gpu, values)
        cuda.memcpy_htod(weights_gpu, weights)

        # Set up result
        res_cpu = np.zeros(size, dtype=self.FLOAT_TYPE)
        res_gpu = cuda.mem_alloc(res_cpu.nbytes)

        args = [
            lat_gpu,
            long_gpu,
            val_gpu,
            weights_gpu,
            np.int32(latitudes.shape[0]),
            res_gpu,
            np.int32(res_cpu.shape[1]),
            np.int32(res_cpu.shape[0]),
            self.FLOAT_TYPE(lat_range[0]),
            self.FLOAT_TYPE(long_range[0]),
            self.FLOAT_TYPE(lat_range[1]),
            self.FLOAT_TYPE(long_range[1]),
        ] + self.get_other_args(**kwargs)

        # Run kernel
        self.__kernel(*args, block=self.BLOCK_SIZE, grid=self.__get_grid(size))

        # Fetch result
        cuda.memcpy_dtoh(res_cpu, res_gpu)
        return res_cpu
