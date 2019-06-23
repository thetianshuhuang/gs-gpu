import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from matplotlib import pyplot as plt
import numpy as np


FLOAT_TYPE = np.float64
FLOAT_CTYPE = "double"

BLOCK_SIZE = (16, 16, 1)


def __get_grid(size):
    return (
        int(np.ceil(size[0] / BLOCK_SIZE[0])),
        int(np.ceil(size[1] / BLOCK_SIZE[1])),
        1)


mod_idw_wf = """
__device__ {f} weight_function({f} x1, {f} y1, {f} x2, {f} y2) {{

    {f} sum = pow(x1 - x2, 2) + pow(y1 - y2, 2);

    if(sum < 0.000001) {{ return 1000000; }}
    else {{ return rsqrt(sum); }}
}}
"""

mod_interp = """
__global__ void idw(
        {f} *x_coord, {f} *y_coord, {f} *val, int size,
        {f} *res, int width_px, int height_px,
        {f} lat_min, {f} long_min, {f} lat_max, {f} long_max)
{{
    int px_x = threadIdx.x + blockIdx.x * blockDim.x;
    int px_y = threadIdx.y + blockIdx.y * blockDim.y;

    if((px_x < width_px) && (px_y < height_px)) {{

        {f} x = long_min + (long_max - long_min) / width_px * px_x;
        {f} y = lat_min + (lat_max - lat_min) / height_px * px_y;

        {f} acc;
        {f} total_weight;

        for(int i = 0; i < size; i++) {{
            {f} weight = weight_function(x_coord[i], y_coord[i], x, y);
            total_weight += weight;
            acc += weight * val[i];
        }}

        res[px_x * width_px + px_y] = acc / total_weight;
    }}
}}
"""

idw_kernel = SourceModule(
    (mod_idw_wf + mod_interp).format(f=FLOAT_CTYPE)).get_function("idw")


def idw(
        lat_cpu, long_cpu, val_cpu, size=[1000, 1000],
        lat_range=[0, 1], long_range=[0, 1]):

    lat_gpu = cuda.mem_alloc(lat_cpu.nbytes)
    long_gpu = cuda.mem_alloc(long_cpu.nbytes)
    val_gpu = cuda.mem_alloc(val_cpu.nbytes)
    cuda.memcpy_htod(lat_gpu, lat_cpu)
    cuda.memcpy_htod(long_gpu, long_cpu)
    cuda.memcpy_htod(val_gpu, val_cpu)

    res_cpu = np.zeros(size, dtype=FLOAT_TYPE)
    res_gpu = cuda.mem_alloc(res_cpu.nbytes)

    assert(
        (lat_cpu.shape[0] == long_cpu.shape[0]) and
        (lat_cpu.shape[0] == val_cpu.shape[0]))

    idw_kernel(
        lat_gpu, long_gpu, val_gpu, np.int32(lat_cpu.shape[0]),
        res_gpu, np.int32(res_cpu.shape[0]), np.int32(res_cpu.shape[1]),
        FLOAT_TYPE(lat_range[0]), FLOAT_TYPE(long_range[0]),
        FLOAT_TYPE(lat_range[1]), FLOAT_TYPE(long_range[1]),
        block=BLOCK_SIZE, grid=__get_grid(size))

    cuda.memcpy_dtoh(res_cpu, res_gpu)

    return res_cpu


if __name__ == '__main__':

    coords = [
        np.array([0, 1, 2, 3, 4, 5], dtype=FLOAT_TYPE),
        np.array([0, 0, 0, 0, 0, 0], dtype=FLOAT_TYPE)
    ]

    values = np.array([0, 1, 2, 3, 4, 5], dtype=FLOAT_TYPE)

    plt.imshow(idw(
        coords[0], coords[1], values,
        size=[256, 256], lat_range=[-3, 3], long_range=[-2, 7]))
    plt.show()
