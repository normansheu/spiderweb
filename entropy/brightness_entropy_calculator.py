import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the matrix from a .npy file
def load_matrix(file_path):
    print("Loading the matrix...")
    matrix = np.load(file_path)
    if matrix.size == 0:
        raise ValueError("The matrix data is empty.")
    print(f"Loaded matrix with shape {matrix.shape}.")
    return matrix

# Step 2: Subdivide the space and calculate average brightness levels
def calculate_brightness_levels(matrix, num_subdivisions, num_levels=10):
    print(f"Calculating brightness levels with {num_subdivisions} subdivisions...")
    
    # Get the dimensions of the matrix
    depth, height, width = matrix.shape
    
    # Calculate the size of each subdivision (block)
    block_depth = depth // num_subdivisions
    block_height = height // num_subdivisions
    block_width = width // num_subdivisions
    
    # Initialize array to store average brightness levels per block
    brightness_levels = np.zeros((num_subdivisions, num_subdivisions, num_subdivisions), dtype=float)
    
    # Calculate average brightness levels for each block
    for i in range(num_subdivisions):
        for j in range(num_subdivisions):
            for k in range(num_subdivisions):
                block = matrix[i * block_depth:(i + 1) * block_depth,
                               j * block_height:(j + 1) * block_height,
                               k * block_width:(k + 1) * block_width]
                brightness_levels[i, j, k] = np.mean(block)
                if np.random.random() < 0.01:
                    print(f"Block ({i}, {j}, {k}): {brightness_levels[i, j, k]}")
    
    # Normalize brightness levels to be between 0 and 1
    brightness_levels = brightness_levels / 255
    
    # Scale brightness levels to fit within [0, num_levels - 1]
    brightness_levels = (brightness_levels * num_levels).astype(int)
    
    return brightness_levels

# Step 3: Record the distribution of the brightness levels
def record_distribution(brightness_levels, num_levels=10):
    print("Recording the distribution of brightness levels...")
    
    # Flatten the brightness levels array
    brightness_levels_flat = brightness_levels.flatten()
    
    # Count how many blocks fall into each brightness level
    distribution = np.zeros(num_levels, dtype=int)
    for level in brightness_levels_flat:
        distribution[level] += 1
    
    print(f"Distribution of brightness levels: {distribution}")
    return distribution

# Step 4: Compute entropy based on the distribution of brightness levels
def compute_entropy(distribution, exclude_zero=False):
    print(f"Computing entropy (excluding level 0: {exclude_zero})...")
    
    # Normalize the distribution to get probabilities
    if exclude_zero:
        distribution = distribution[1:]  # Exclude level 0
    probabilities = distribution / distribution.sum()
    
    # Compute entropy
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    print(f"Computed entropy: {entropy}")
    return entropy

# Step 5: Calculate and record entropies for different subdivisions
def calculate_entropies_for_subdivisions(matrix, min_subdivisions=30, max_subdivisions=50):
    entropies_including_zero = []
    entropies_excluding_zero = []
    
    for num_subdivisions in range(min_subdivisions, max_subdivisions + 1):
        brightness_levels = calculate_brightness_levels(matrix, num_subdivisions)
        distribution = record_distribution(brightness_levels)
        entropy_including_zero = compute_entropy(distribution, exclude_zero=False)
        entropy_excluding_zero = compute_entropy(distribution, exclude_zero=True)
        entropies_including_zero.append((num_subdivisions, entropy_including_zero))
        entropies_excluding_zero.append((num_subdivisions, entropy_excluding_zero))
    
    return entropies_including_zero, entropies_excluding_zero

if __name__ == "__main__":
    # Example usage
    file_path = '../video_processing/point_clouds/@013 255 2024-10-05 03-18-53 brightness.npy'
    matrix = load_matrix(file_path)[:-20, :, :650]
    entropies_including_zero, entropies_excluding_zero = calculate_entropies_for_subdivisions(matrix)
    
    # Plot the results
    subdivisions, entropies_inc = zip(*entropies_including_zero)
    _, entropies_exc = zip(*entropies_excluding_zero)
    
    plt.figure()
    plt.plot(subdivisions, entropies_inc, label='Including Level 0')
    plt.plot(subdivisions, entropies_exc, label='Excluding Level 0')
    plt.xlabel('Number of Subdivisions')
    plt.ylabel('Entropy')
    plt.legend()
    plt.title('Entropy vs. Number of Subdivisions')
    plt.show()