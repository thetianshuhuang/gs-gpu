"""Image overlay routines"""

import folium
from folium import raster_layers
from folium.plugins import FastMarkerCluster

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
        weights=None,
        size=(1024, 1024),
        lat_range=(0, 1),
        long_range=(0, 1),
        truncate=0.05,
        hue_range=(0, 0.3),
        **kwargs):
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
    weights : np.array
        Weights array; defaults to all 1
    size : int[2]
        Image size to generate
    lat_range : float[2]
        Latitude range
    long_range : float[2]
        Longitude range
    truncate : float
        Truncate argument for to_image.
    hue_range : float
        Truncate argument for to_image.
    **kwargs : dict
        args to pass to overlay_image.

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
        size=size, lat_range=lat_range, long_range=long_range, weights=weights)

    image = to_image(interp, truncate=truncate, hue_range=hue_range)

    map_ = overlay_image(
        image, latitudes, longitudes,
        lat_range=lat_range, long_range=long_range, **kwargs)

    return map_, interp, image


def overlay_image(
        image, latitudes, longitudes,
        lat_range=(0, 1), long_range=(0, 1), opacity=0.6,
        marker_size=1, use_fmc=True, tiles='Stamen Terrain'):
    """Overlay an image onto a folium map.

    Parameters
    ----------
    image : np.array
        Image array to overlay
    latitudes : np.array
        Latitude array.
    longitudes : np.array
        Longitude array
    marker_size : float
        Size of markers. If marker_size=0, no markers are shown.
    tiles : str
        Tile type to use for folium map. Defaults to 'Stamen Terrain'.
    use_fmc : bool
        If True, uses FastMarkerClusters to avoid jupyter notebook limitations.
        WARNING: if you set use_fmc to false, large datasets (>1000 points) may
        cause the map to refuse to display on Jupyter notebooks.
    lat_range : float[2]
        Latitude range
    long_range : float[2]
        Longitude range
    """

    # Base Map
    map_ = folium.Map(
        location=(
            (lat_range[0] + lat_range[1]) / 2,
            (long_range[0] + long_range[1]) / 2),
        tiles=tiles,
        zoom_start=8)

    # Interpolation Overlay
    map_.add_child(raster_layers.ImageOverlay(
        image, opacity=opacity,
        bounds=[
            [lat_range[0], long_range[0]],
            [lat_range[1], long_range[1]]]
    ))

    # Markers
    if marker_size > 0:
        if use_fmc:
            FastMarkerCluster(
                data=list(zip(latitudes, longitudes))).add_to(map_)
        else:
            for x, y in zip(latitudes, longitudes):
                folium.CircleMarker((x, y), radius=marker_size).add_to(map_)

    return map_
