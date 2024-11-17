import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_pcd_array,
    calculate_density_levels,
    record_distribution,
)


current_path = pathlib.Path(__file__).resolve().parent

# Compute entropy based on the distribution of density levels
def compute_entropy(distribution, exclude_zero=False):
    print(f"Computing entropy (excluding level 0: {exclude_zero})...")
    
    # Normalize the distribution to get probabilities
    if exclude_zero:
        distribution = distribution[1:]  # Exclude level 0
    probabilities = distribution / distribution.sum()
    
    # Compute entropy
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    print(f"Entropy: {entropy}")
    return entropy


def calculate_entropies_for_subdivisions(points, num_levels=10, min_M=100, max_M=120):
    print(f"Calculating entropies for subdivisions from {min_M} to {max_M}...")
    entropies_including_zero = []
    entropies_excluding_zero = []
    averages_excluding_zero = []
    
    for num_M in range(min_M, max_M + 1):
        density, density_levels = calculate_density_levels(points, M=num_M, num_levels=num_levels)
        distribution = record_distribution(density_levels)
        entropy_including_zero = compute_entropy(distribution, exclude_zero=False)
        entropy_excluding_zero = compute_entropy(distribution, exclude_zero=True)
        entropies_excluding_zero.append(entropy_excluding_zero)
        # Append the sliding average of entropies of previous 5
        averages_excluding_zero.append(np.mean(entropies_excluding_zero[-20:]))
        entropies_including_zero.append(entropy_including_zero)
    
    return entropies_including_zero, entropies_excluding_zero, averages_excluding_zero


def plot_entropies_vs_subdivisions(points, num_levels=10, min_M=100, max_M=120):
    entropy_with_0, entropy_without_0, avg_entropy_without_0 = calculate_entropies_for_subdivisions(points, num_levels=num_levels, min_M=min_M, max_M=max_M)
    print(f"Entropy with 0: {entropy_with_0}")
    print(f"Entropy without 0: {entropy_without_0}")
    print(f"Average entropy without 0: {avg_entropy_without_0}")

    # Plot entropy vs number of subdivisions
    subdivisions = range(min_M, max_M + 1)

    plt.figure()
    plt.plot(subdivisions, entropy_with_0, label='Including Level 0')
    plt.plot(subdivisions, entropy_without_0, label='Excluding Level 0')
    plt.plot(subdivisions, avg_entropy_without_0, label='Average excluding Level 0')
    plt.xlabel("Number of Subdivisions (M)")
    plt.ylabel("Entropy")
    plt.legend()
    plt.title("Entropy vs Number of Subdivisions")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute entropy of a point cloud.")
    parser.add_argument("--file_name", type=str, help="File name of the pcd file of a web (without extension .npy).")
    parser.add_argument("--num_M", type=int, help="Number of subdivisions for density calculation.")
    parser.add_argument("--num_levels", type=int, default=10, help="Number of density levels. Default is 10.")
    parser.add_argument("--min_M", type=int, default=20, help="Minimum number of subdivisions. Default is 20.")
    parser.add_argument("--max_M", type=int, default=120, help="Maximum number of subdivisions. Default is 120.")
    args = parser.parse_args()

    path = f"../point_clouds/{args.file_name}.npy"
    num_M = args.num_M
    num_levels = args.num_levels
    min_M = args.min_M
    max_M = args.max_M

    pcd = load_pcd_array(path)
    density, density_levels = calculate_density_levels(pcd, M=num_M, num_levels=num_levels)
    distribution_levels = record_distribution(density_levels)
    entropy = compute_entropy(distribution_levels, exclude_zero=True)
    print(f"Entropy without 0: {entropy}")

    plot_entropies_vs_subdivisions(pcd, num_levels=num_levels, min_M=min_M, max_M=max_M)
