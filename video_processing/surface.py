"""
Surface reconstruction from point cloud
Based on Open3D's features: alpha shape, ball pivoting, Poisson reconstruction
https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html#Surface-reconstruction
"""
import argparse
import open3d as o3d


def alpha_shapes(pcd):
    alpha = 5.0
    print(f"alpha shape, {alpha=}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def ball_pivoting(pcd):
    # This may take more than 10 minutes
    radii = [5.0, 10.0, 30.0, 50.0, 100.0]
    print(f"Ball pivoting, {radii=}")
    print("Estimate normal vectors...")
    pcd.estimate_normals()
    print("Estimation completed!")
    print("Apply ball pivoting...")
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])


# TODO: Need to change values below to make this work
def poisson(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        print(mesh)
        o3d.visualization.draw_geometries(
            [mesh],
            zoom=0.664,
            front=[-0.4761, -0.4698, -0.7434],
            lookat=[1.8900, 3.2596, 0.9284],
            up=[0.2304, -0.8825, 0.4101],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the .pcd file")
    parser.add_argument("--method", choices=["alpha", "ball", "poisson"], default="alpha", help="Surface reconstruction method")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.path)
    if args.method == "alpha":
        alpha_shapes(pcd)
    elif args.method == "ball":
        ball_pivoting(pcd)
    elif args.method == "poisson":
        poisson(pcd)
