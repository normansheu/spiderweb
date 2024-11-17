import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt

from utils import load_pcd, calculate_density


def p_average(density, p=1):
    print(f"Inferring density for each block based on its neighbors as p-power mean with p = {p:.4f}")
    
    # Kernel for averaging densities of 6 neighboring blocks
    kernel = np.zeros((3, 3, 3))
    kernel[0, 1, 1] = 1/6
    kernel[2, 1, 1] = 1/6
    kernel[1, 0, 1] = 1/6
    kernel[1, 2, 1] = 1/6
    kernel[1, 1, 0] = 1/6
    kernel[1, 1, 2] = 1/6

    if p > 0:
        density_p = np.power(density, p)
        print(density_p.shape, kernel.shape)
        inferred_density = scipy.signal.convolve(density_p, kernel, mode="same") ** (1/p)  # Same shape as input
    else:
        # Geometry mean when p = 0, exp(avg(log(density)))
        eps = 1e-6  # Small value to avoid log(0)
        density_p = np.log(density + eps)
        inferred_density = np.exp(scipy.signal.convolve(density_p, kernel, mode="same"))
    return inferred_density


def p_harmonicity(density, p=1, q=1):
    """Error between density and p-averaged density, ues averaged L^q norm"""
    assert q > 0, "q must be positive"
    depth, height, width = density.shape
    p_average_density = p_average(density, p)
    err = np.abs(density - p_average_density)[1: depth - 1, 1: height - 1, 1: width - 1]
    return np.mean(err ** q) ** (1/q)


def plot_error_vs_p(density, q=1, delta=0.1, max_p=3.0):
    p_range = [delta * i for i in range(0, int(max_p / delta) + 1)]
    errors = [p_harmonicity(density, p, q) for p in p_range]
    best_p = p_range[np.argmin(errors)]
    print(f"Best p: {best_p}")
    plt.plot(p_range, errors)
    plt.xlabel("p")
    plt.ylabel(f"L^{q} error")
    plt.title(f"L^{q} error vs p")
    plt.show()


def visualize_inferred_vs_actual(actual_density, inferred_density, slice_idx=66):
    # Compare a slice of actual and inferred densities
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot actual densities for a specific slice
    axes[0].imshow(actual_density[slice_idx, :, :], cmap='Greys', interpolation='nearest')
    axes[0].set_title("Actual Density (Slice)")
    axes[0].axis('off')
    
    # Plot inferred densities for the same slice
    axes[1].imshow(inferred_density[slice_idx, :, :], cmap='Greys', interpolation='nearest')
    axes[1].set_title("Inferred Density (Slice)")
    axes[1].axis('off')
    
    plt.suptitle(f"Comparison of Actual and Inferred Densities at Depth Slice {slice_idx}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer density from neighboring blocks.")
    parser.add_argument("--file_name", type=str, help="File name of the pcd file of a web (without extension).")
    parser.add_argument("--num", type=int, default=20, help="Number of subdivisions for density calculation. Default is 20.")
    parser.add_argument("--p", type=float, default=1.0, help="Power for p-power mean. Default is 1.0.")
    parser.add_argument("--q", type=float, default=1.0, help="Power for L^q norm. Default is 1.0.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the inferred density vs original density.")
    parser.add_argument("--plot_p", action="store_true", help="Plot error vs p. Also find the best p.")
    args = parser.parse_args()

    path = f"../point_clouds/{args.file_name}.pcd"
    num = args.num
    p = args.p
    q = args.q

    points = load_pcd(path)
    density = calculate_density(points, num_subdivisions=num)
    inferred_density = p_average(density, p)

    if args.visualize:
        visualize_inferred_vs_actual(density, inferred_density)
    if args.plot_p:
        plot_error_vs_p(density, q=q)
