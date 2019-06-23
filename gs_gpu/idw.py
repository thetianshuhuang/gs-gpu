"""Inverse distance weighted interpolation"""

from .interpolate import BaseInterpolation


_MOD_IDW_WEIGHT = """
__device__ {f} weight_function({f} x1, {f} y1, {f} x2, {f} y2) {{

    {f} sum = pow(x1 - x2, 2) + pow(y1 - y2, 2);

    if(sum < 0.000001) {{ return 1000000; }}
    else {{ return rsqrt(sum); }}
}}
"""


class IDWInterpolation(BaseInterpolation):
    WEIGHT_FUNCTION = _MOD_IDW_WEIGHT
