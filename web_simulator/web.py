from __future__ import annotations
from typing import List, Tuple, Optional

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import collections as mc


class Point3D:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def dist(self, other: Point3D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def __repr__(self) -> str:
        return f"Point3D({self.x},{self.y},{self.z})"

    def __eq__(self, other: Point3D) -> bool:
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float, float, float]:
        return [self.x, self.y, self.z]


VERTICES_T = List[Point3D]
EDGE_T = Tuple[int, int]
EDGES_T = List[EDGE_T]

class Web:
    def __init__(self, vertices: VERTICES_T = [], edges: EDGES_T = [], name: str = "") -> None:
        self._vertices: VERTICES_T = vertices
        self._edges: EDGES_T = edges
        self.name = name

    @property
    def vertices(self) -> VERTICES_T:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: VERTICES_T):
        self._vertices = vertices

    @property
    def edges(self) -> EDGES_T:
        return self._edges

    @edges.setter
    def edges(self, edges: EDGES_T):
        self.edges = edges

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_df(self) -> pd.DataFrame:
        """Transform web data into pandas dataframe.
        It has the following form:

        | p1.x | p1.y | p1.z |
        | q1.x | q1.y | q1.z |
        | None | None | None |
        | p2.x | p2.y | p2.z |
        | q2.x | q2.y | q2.z |
        | None | None | None |
        ...

        Returns:
            pd.DataFrame: DataFrame with edge information.
        """
        rows = []
        edges = self.edges
        vertices = self.vertices
        for s, e in edges:
            rows.append([vertices[s].x, vertices[s].y, vertices[s].z])
            rows.append([vertices[e].x, vertices[e].y, vertices[e].z])
            rows.append([np.nan, np.nan, np.nan])
        df = pd.DataFrame(rows, columns=["x", "y", "z"])
        return df

    @staticmethod
    def from_df(self) -> Web:
        pass

    def plot(self) -> None:
        df = self.to_df()
        fig = px.line_3d(df, x='x', y='y', z='z')
        fig.update_traces(connectgaps=False)
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        fig.show()

    def thick_plane_section(self, center: float, thickness: float) -> Web:
        """Sample more realistic section, which has thickness.
        For given `center` and `thickness`, the thick plane is a region
        whose x-coordinate lies between `center - thickness/2` and `center + thickness/2`.

        Args:
            points (List[Point3D]): Vertices of spider web graph.
            edges (List[Tuple[int, int]]): Edges of spider web graph.
            center (float): Coordinate of the center of the section.
            thickness (float): thickness of section.

        Returns:
            Line segments that are contained in both web and the thick section.
            Tuple of list of vertices and edges.
        """
        smin = center - thickness / 2
        smax = center + thickness / 2

        # auxiliary functions
        def check_intersect(p1: Point3D, p2: Point3D, x_coord: float) -> bool:
            return (p1.x - x_coord) * (p2.x - x_coord) <= 0
        
        def find_intersect(p1: Point3D, p2: Point3D, x_coord: float) -> Optional[Point3D]:
            if check_intersect(p1, p2, x_coord):
                # t * p1.x + (1 - t) * p2.x = x_coord
                # t = (x_coord - p2.x) / (p1.x - p2.x)
                if p1.x == x_coord:
                    return p1
                elif p2.x == x_coord:
                    return p2
                else:
                    t = (x_coord - p2.x) / (p1.x - p2.x)
                    ix = x_coord
                    iy = t * p1.y + (1 - t) * p2.y
                    iz = t * p1.z + (1 - t) * p2.z
                    return Point3D(ix, iy, iz)
            else:
                return None

        def add_vertices_and_edge(vertices: VERTICES_T, edges: EDGES_T, p1: Point3D, p2: Point3D) -> None:
            i1 = None
            i2 = None
            if p1 in new_vertices:
                i1 = new_vertices.index(p1)
            if p2 in new_vertices:
                i2 = new_vertices.index(p2)
            if (i1, i2) == (None, None):  # both points are new
                vertices.append(p1)
                vertices.append(p2)
                edges.append((len(new_vertices) - 2, len(new_vertices) - 1))
            elif i1 is None:  # pei is already in the list
                new_vertices.append(p1)
                new_edges.append((len(new_vertices) - 1, i2))
            elif i2 is None:  # psi is already in the list
                new_vertices.append(p2)
                new_edges.append((len(new_vertices) - 1, i1))
            else:  # both already exists in the list
                new_edges.append((i1, i2))

        new_vertices = []
        new_edges = []
        for s, e in self.edges:
            ps = self.vertices[s]
            pe = self.vertices[e]

            ps_in_section = (smin < ps.x < smax)
            pe_in_section = (smin < pe.x < smax)

            if ps_in_section and pe_in_section:
                # vertices and edges entirely contained in the region
                add_vertices_and_edge(new_vertices, new_edges, ps, pe)
            else:
                p_min = find_intersect(ps, pe, smin)
                p_max = find_intersect(ps, pe, smax)
                if p_min is not None and p_max is None:
                    # intersect with starting section
                    if ps_in_section:
                        add_vertices_and_edge(new_vertices, new_edges, ps, p_min)
                    else:
                        assert pe_in_section
                        add_vertices_and_edge(new_vertices, new_edges, pe, p_min)
                elif p_min is None and p_max is not None:
                    # intersect with ending section
                    if ps_in_section:
                        add_vertices_and_edge(new_vertices, new_edges, ps, p_max)
                    else:
                        assert pe_in_section
                        add_vertices_and_edge(new_vertices, new_edges, pe, p_max)
                elif p_min is not None and p_max is not None:
                    # intersect with both
                    add_vertices_and_edge(new_vertices, new_edges, p_min, p_max)

        return Web(new_vertices, new_edges)

    def plot_thick_plane_section(self, center: float, thickness: float, width: int = 1920, height: int = 1080) -> None:
        """Plot yz-plane projection of a thick plane section.

        Args:
            center (float): Coordinate of the center of the section.
            thickness (float): Width of section.
        """
        # TODO: add Gaussian blur that reflects laser's brightness
        thick_section = self.thick_plane_section(center, thickness)
        sec_df = thick_section.to_df()[["y", "z"]]
        fig = px.line(sec_df, x="y", y="z")
        fig.update_traces(connectgaps=False)
        fig.update_layout(
            xaxis=dict(range=[-width / 2, width / 2], showgrid=False),
            yaxis=dict(range=[0, height], showgrid=False),
        )
        fig.show()

    def save(self, path: str) -> None:
        """Save web as csv files, `path/vertices.csv` and `path/edges.csv`.

        Args:
            path (str): Directory to save the web.
        """
        Path(path).mkdir(exist_ok=True)

        # points
        columns = ["x", "y", "z"]
        coords = [p.to_list() for p in self.vertices]
        df_points = pd.DataFrame(data=coords, columns=columns, dtype=float)
        df_points.to_csv(Path(path) / "vertices.csv")

        # edges
        columns = ["i", "j"]
        arr = [[e[0], e[1]] for e in self.edges]  # (num_edges, 2)
        df_edges = pd.DataFrame(data=arr, columns=columns, dtype=int)
        df_edges.to_csv(Path(path) / "edges.csv")

    @staticmethod
    def load(path: str) -> Web:
        """Load web from csv files.

        Args:
            path (str): Directory to load the web

        Returns:
            Web: Loaded web
        """
        path_vertices = Path(path) / "vertices.csv"
        path_edges = Path(path) / "edges.csv"

        df_vertices = pd.read_csv(path_vertices)
        vertices = df_vertices.values.tolist()
        vertices = [Point3D(p[0], p[1], p[2]) for p in vertices]

        df_edges = pd.read_csv(path_edges)
        edges = df_edges.values.tolist()
        edges = [(e[0], e[1]) for e in edges]

        return Web(vertices, edges)

    def simulate_frames(self, img_w: int = 1920, img_h: int = 1080, depth: int = 1920, num_frames: int = 100, frame_thickness: float = 10.0, save_dir: Optional[str] = None) -> np.ndarray:
        """Generate synthetic frame images of the web.
        
        Args:
            img_w (int, optional): image width. Defaults to 1920.
            img_h (int, optional): image height. Defaults to 1080.
            num_frames (int, optional): number of frames. Defaults to 100.
            frame_thickness (float, optional): width of a single thick section. Defaults to 0.1.

        Returns:    
            np.ndarray: 3D numpy array of shape (num_frames, img_h, img_w), consist of black/white pixel values.
        """
        # plt.ioff()
        y_start = -depth / 2
        y_end = depth / 2
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        for i in range(num_frames):
            y_i = y_start + (i / (num_frames - 1)) * (y_end - y_start)
            section_i = self.thick_plane_section(center=y_i, thickness=frame_thickness)
            sec_df = section_i.to_df()[["y", "z"]]

            # make plot
            lines = []
            num_lines = len(sec_df.index) // 3
            for j in range(num_lines):
                line_j_1 = sec_df.iloc[3 * j].values.flatten().tolist()
                line_j_2 = sec_df.iloc[3 * j + 1].values.flatten().tolist()
                lines.append([line_j_1, line_j_2])

            lc = mc.LineCollection(lines, linewidths=1.0, colors='1')
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig, ax = plt.subplots(figsize=(img_w * px, img_h * px), frameon=False)
            # ax = fig.add_axes(frameon=False)
            ax.add_collection(lc)
            ax.set_xlim([-img_w / 2, img_w / 2])
            ax.set_ylim([0, img_h])
            ax.set_aspect('equal')
            ax.set_facecolor('0')
            _ = [s.set_visible(False) for s in ax.spines.values()]
            ax.margins(0.0)

            fig.patch.set_facecolor('0')
            fig.add_axes(ax)
            fig.subplots_adjust(
                top=1.0,
                bottom=0.0,
                left=0.0,
                right=1.0,
                hspace=0.0,
                wspace=0.0,
            )  # remove all the margins outside frame

            # save
            path_i = save_dir / f"{self.name}_{i}.png"
            fig.savefig(path_i)
