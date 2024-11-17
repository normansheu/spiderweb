## Computation of entropy of silk density

### `utils.py`

### `compute_entropy.py`

Compute entropy of a single web.

### `inference_from_neighbors.py`

Use p-average to infer a silk density from neighbors. Compute (averaged) L^q-error.
Plot a graph of errors in `p` (for a fixed `q`) and find `p` with minimum inference error.

Arguments:

- `file_name`: File is assumed to be in the `spiderweb/point_clouds` directory.
- `num`
- `p`
- `q`
- `visualize` (boolean flag)
- `plot_p` (boolean flag)

Example:
```
python3 inference_from_neighbors.py --file_name "@011 255 2024-10-04 03-20-37" --num 20 --p 1.0 --q 1.0 --plot_p
```

