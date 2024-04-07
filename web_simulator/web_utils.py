from __future__ import annotations
import random
from typing import Callable, Union
import numpy as np

from web import Point3D, VERTICES_T, EDGES_T, Web


def sample_segment(p1: Point3D, p2: Point3D, threshold_min: float = 1.0, threshold_max: Union[float, np.infty] = np.infty):
    d = p1.dist(p2)
    if d < threshold_min:
        return True
    elif d > threshold_max:
        return False
    else:
        # probability is inversely proportional to the distance
        # may change this function if needed
        # alpha = 2
        beta = np.log(threshold_max / threshold_min) / np.log(100.0)
        alpha = threshold_min ** beta
        prob = alpha / (d ** beta)
        return random.random() < prob


def sample_vertices(func: Callable, x_len: float, y_len: float, N: int = 1000, epsilon: float = 10.0) -> VERTICES_T:
    """Sample vertices on a graph of a given function with random noise.

    Args:
        func (Callable): 2-variable function whose graph would simulate web.
        x_len (float): width of floor.
        y_len (float): height of floor.
        N (int, optional): Number of points to be sampled. Defaults to 1000.
        epsilon (float, optional): Error bound for noise. Defaults to 10.0.

    Returns:
        VERTICES_T: _description_
    """
    x_coords = np.random.uniform(-x_len / 2, x_len / 2, size=N)
    y_coords = np.random.uniform(-y_len / 2, y_len / 2, size=N)
    z_coords = func(x_coords, y_coords) + np.random.uniform(-epsilon, epsilon)

    vertices = [Point3D(x_, y_, z_) for x_, y_, z_ in zip(x_coords, y_coords, z_coords)]
    return vertices


def sample_edges(vertices: VERTICES_T, threshold_min: float = 1.0, threshold_max: Union[float, np.infty] = np.infty) -> EDGES_T:
    # return pair of indices corresponding to the sampled edges
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if sample_segment(vertices[i], vertices[j], threshold_min=threshold_min, threshold_max=threshold_max):
                edges.append((i, j))
    return edges


def sample_web(
    func: Callable,
    x_len: float,
    y_len: float,
    N: int = 1000,
    epsilon: float = 0.01,
    threshold_min: float = 1.0,
    threshold_max: Union[float, np.infty] = np.infty,
) -> Web:
    vertices = sample_vertices(func, x_len, y_len, N, epsilon)
    edges = sample_edges(vertices, threshold_min, threshold_max)
    return Web(vertices, edges)
