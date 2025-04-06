import open3d as o3d

def visualize_pcd(file_path):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Check if point cloud data is loaded correctly
    if pcd.is_empty():
        print("Error: Point cloud data is empty or could not be loaded.")
        return
    
    # Display the point cloud
    o3d.visualization.draw_geometries([pcd])

# Replace 'your_file.pcd' with the path to your PCD file
file_path = "video_processing/point_clouds/sparse3 255 2024-11-30 11-29-33.pcd"
visualize_pcd(file_path)

