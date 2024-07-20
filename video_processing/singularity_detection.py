import argparse
import os
from timeit import default_timer

import numpy as np
import open3d as o3d
import pandas as pd
import plotly.express as px

import hades


# New plot_filt function with plotly
def plot_filt(X, filt):
    nfilt = np.logical_not(filt)
    Xf = X[filt]
    Xnf = X[nfilt]
    print(f"{Xf.shape=}, {Xnf.shape=}")

    dim = X.shape[1]
    if dim == 2:
        columns = ["x", "y", "filter"]
    else:
        columns = ["x", "y", "z", "filter"]
    if len(Xf) > 0:
        sm_col = np.ones((Xf.shape[0], 1))
        Xf_ = np.concatenate([Xf, sm_col], axis=1)
        df_Xf = pd.DataFrame(Xf_, index=None, columns=columns)
    if len(Xnf) > 0:
        sing_col = np.zeros((Xnf.shape[0], 1))
        Xnf_ = np.concatenate([Xnf, sing_col], axis=1)
        df_Xnf = pd.DataFrame(Xnf_, index=None, columns=columns)
    
    df = pd.concat([df_Xf, df_Xnf], axis=0)
    # print(df.head)
    
    if dim == 2:
        fig = px.scatter(df, x="x", y="y", color="filter")
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="filter")
    fig.update_traces(marker_size=5)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the .pcd file")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.path)
    pcd_arr = np.asarray(pcd.points)
    pcd_arr /= 500.0  # normalize
    print(f"Number of points: {pcd_arr.shape[0]}")

    N = 50000
    pcd_arr = pcd_arr[:N]
    print("Start detection")
    st = default_timer()
    verdict = hades.judge(pcd_arr)
    et = default_timer()
    print(f"Detection finished in {et - st:.2f}s")

    fig = plot_filt(pcd_arr, verdict['label'])
    dir = os.path.dirname(args.path)
    filename = os.path.basename(args.path)
    # Blue points (i.e. value = 0) are singular points
    fig.write_html(os.path.join(dir, f"{filename[:-4]}.html"))
