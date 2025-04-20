# import torch

# def inspect_data(file_path):
#     # Load the data
#     data = torch.load(file_path)
    
#     # Print basic information about the data structure
#     print(f"Number of elements in the data tuple: {len(data)}")
#     print("\nDetailed structure:")
    
#     for i, item in enumerate(data):
#         if torch.is_tensor(item):
#             print(f"Element {i}: Tensor of shape {item.shape} with dtype {item.dtype}")
#         else:
#             print(f"Element {i}: {type(item)} - {str(item)[:100]}...")
    
#     # Get the input tensor (assuming it's the first element)
#     input_data = data[0]
#     print("\nSample information:")
#     print(f"Number of samples: {input_data.shape[0]}")
#     print(f"Nodes per sample: {input_data.shape[1]}")
#     print(f"Features per node: {input_data.shape[2]}")
    
#     # Check actual node counts in samples
#     print("\nActual node counts in first 10 samples:")
#     for i in range(min(10, input_data.shape[0])):
#         # Count non-zero nodes (assuming nodes with all-zero features are padding)
#         active_nodes = torch.any(input_data[i] != 0, dim=1).sum().item()
#         print(f"Sample {i}: {active_nodes} active nodes (of {input_data.shape[1]} possible)")

# # Example usage
# if __name__ == "__main__":
#     file_path = "video_processing/dataset_webs_medium.pt"
#     inspect_data(file_path)

import torch

def print_sample_details(file_path, sample_idx=0):
    # Load the data
    data = torch.load(file_path)
    
    # Extract the first sample
    sample = data[0][sample_idx]  # Shape [64, 16]
    properties = data[1][sample_idx]  # Shape [7]
    
    print("=== Sample Details ===")
    print(f"Sample index: {sample_idx}")
    print(f"Actual node count: {int(properties[4].item())}")
    print(f"Actual edge count: {int(properties[5].item())}")
    print(f"Average degree: {properties[6].item():.2f}\n")
    
    print("=== Node Features ===")
    print("Format for each node: [node_id, x, y, z, neighbor1, neighbor2, ..., neighbor6, ...other_features]")
    print("(Note: Zero values typically indicate padding)\n")
    
    # Print first 10 nodes (or all if less than 10)
    for node_idx in range(sample.shape[0]):
        node_data = sample[node_idx]
        print(f"Node {node_idx}:")
        print(f"  Coordinates: ({node_data[1]:.3f}, {node_data[2]:.3f}, {node_data[3]:.3f})")
        print(f"  Neighbors: {[int(x.item()) for x in node_data[4:10] if x.item() != 0]}")
        print(f"  Other features: {[f'{x:.3f}' for x in node_data[10:]]}")
        print()
    
    print("\n=== Global Properties ===")
    property_names = data[3]
    for name, value in zip(property_names, properties):
        print(f"{name}: {value.item():.3f}")

if __name__ == "__main__":
    file_path = "video_processing/dataset_webs_medium.pt"
    print_sample_details(file_path, sample_idx=0)