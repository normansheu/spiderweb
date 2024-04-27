# Web Simulator


Generate a systhetic web. First, we define the underlying bivariate function $z = f(x, y)$, and the points (vertices) will be sampled on the graph of the function with some random noises.
Then we randomly connect some pair of points according to the distances between them.
You can try `web_simulator_example.ipynb` to generate a synthetic web on a parabola.

## `web.py`

Defines a class `Point3D` and `Web`, where the latter is essentially a 3D graph with coordinates of the points and indices of the end vertices of the edges.
The most important function is `Web.simulate_frames`, which simulate the scanning process of a real web and save the image of thick sections.
The results are greyscale images with white colors for the web.

## `web_utils.py`

Utility functions for sampling vertices and edges. To sample vertices, we randomly choose points on the graph of the function, then add a (uniform) noise to them.
To sample edges, we loop over all pairs of points, and connect them randomly, where the probability to connect decreases as the distance between the two points increase (see `sample_segment` for details).
