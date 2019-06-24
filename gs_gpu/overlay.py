"""Image overlay routines"""

import folium
from folium import raster_layers

from matplotlib import colors
import numpy as np


def to_image(array, truncate=0.05, hue_range=(0, 1)):
    """Convert float array to image, with hue proportional to value

    Parameters
    ----------
    array : np.array
        Input array. Should be 2-dimensional.
    truncate : float
        Percent of the data to leave out on both ends. Values in the top and
        bottom ``truncate`` of the data are set as the max hue and min hue,
        respectively
    hue_range : float[2]
        Range of hue values for the HSV color scale. Scales from 0-1.

    Returns
    -------
    np.array
        RGB array, with values in the range (0, 255).

    """

    # Get 5th and 95th percentile (or something else)
    lower = np.percentile(array.reshape(-1, 1), truncate * 100)
    upper = np.percentile(array.reshape(-1, 1), (1 - truncate) * 100)

    # Scale
    array = (array - lower) / (upper - lower)

    # Truncate
    array[array < 0] = 0
    array[array > 1] = 1

    # Set hue range
    array = (array + hue_range[0]) * (hue_range[1] - hue_range[0])

    # Make array
    return (
        255 * colors.hsv_to_rgb(
            np.dstack((array, np.ones_like(array), np.ones_like(array)))
        )
    ).astype(np.uint8)


def interpolate_overlay(
        kernel, latitudes, longitudes, values,
        size=(1024, 1024),
        lat_range=(0, 1), long_range=(0, 1),
        opacity=0.6,
        truncate=0.05, hue_range=(0, 0.3)):
    """Create interpolation overlay. All input arrays should match the type
    that the kernel was initialized with.

    Parameters
    ----------
    kernel : BaseInterpolation
        Interpolation kernel to use (i.e. IDWInterpolation())
    latitudes : np.array
        Latitude array.
    longitudes : np.array
        Longitude array
    values : np.array
        Values array
    truncate : float
        Truncate argument for to_image.
    hue_range : float
        Truncate argument for to_image.

    Returns
    -------
    [folium.Map, np.array, np.array]
        [0] Created folium map. View by calling it's __repr__ (just write it
            on its own in a line)
        [1] Computed interpolation (float)
        [2] Converted image (uint8_t RGB)
    """

    interp = kernel.interpolate(
        latitudes, longitudes, values,
        size=size, lat_range=lat_range, long_range=long_range)

    image = to_image(interp, truncate=truncate, hue_range=hue_range)

    map_ = folium.Map(
        location=(
            (lat_range[0] + lat_range[1]) / 2,
            (long_range[0] + long_range[1]) / 2),
        tiles='Stamen Terrain',
        zoom_start=8)

    map_.add_child(raster_layers.ImageOverlay(
        image, opacity=opacity,
        bounds=[
            [lat_range[0], long_range[0]],
            [lat_range[1], long_range[1]]]
    ))

    return map_, interp, image