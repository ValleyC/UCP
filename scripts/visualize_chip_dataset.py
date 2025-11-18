"""
Visualization Script for Chip Placement Datasets
Loads and visualizes generated chip placement instances
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pickle
import argparse
from PIL import Image, ImageDraw
from pathlib import Path


def hsv_to_rgb(h, s, v):
    """Convert HSV color to RGB"""
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)


def canvas_to_pixel(coords, width, height):
    """
    Convert normalized canvas coordinates [-1, 1] to pixel coordinates

    Args:
        coords: (N, 2) tensor or numpy array
        width, height: image dimensions

    Returns:
        pixel_coords: (N, 2) array
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()

    pixel_x = (0.5 + coords[:, 0] / 2) * width
    pixel_y = (0.5 - coords[:, 1] / 2) * height  # Y-axis flipped
    return np.stack([pixel_x, pixel_y], axis=1)


def visualize_chip_placement(positions, data, width=2048, height=2048, plot_edges=True):
    """
    Visualize chip placement with components and netlist

    Args:
        positions: (V, 2) numpy array of component center positions
        data: PyTorch Geometric Data object with:
            - x: (V, 2) component sizes
            - edge_index: (2, E)
            - edge_attr: (E, 4) terminal offsets
            - is_ports: (V,) boolean mask
        width, height: output image size
        plot_edges: whether to draw netlist connections

    Returns:
        image: numpy array (H, W, 3)
    """
    # Create image
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Convert to tensors if needed
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions)
    sizes = data.x
    mask = data.is_ports if hasattr(data, 'is_ports') else None

    V = positions.shape[0]
    h_step = 1 / V

    # Compute bounding boxes
    left_bottom = positions - sizes / 2
    right_top = positions + sizes / 2

    # Check if components are within bounds
    inbounds = torch.logical_and(left_bottom >= -1, right_top <= 1)
    inbounds = torch.logical_and(inbounds[:, 0], inbounds[:, 1])

    # Convert to pixel coordinates
    left_bottom_px = canvas_to_pixel(left_bottom.numpy(), width, height)
    right_top_px = canvas_to_pixel(right_top.numpy(), width, height)

    # Draw components
    for i in range(V):
        color = hsv_to_rgb(
            i * h_step,
            1.0 if (mask is None or not mask[i]) else 0.2,
            0.9 if inbounds[i] else 0.5
        )

        # Draw rectangle (left, top, right, bottom)
        draw.rectangle(
            [left_bottom_px[i, 0], right_top_px[i, 1],
             right_top_px[i, 0], left_bottom_px[i, 1]],
            fill=color,
            outline=None
        )

    # Draw netlist edges
    if plot_edges and hasattr(data, 'edge_attr'):
        # Get unique edges (undirected)
        unique_edges = data.edge_attr.shape[0] // 2
        edge_index_unique = data.edge_index[:, :unique_edges]
        edge_attr_unique = data.edge_attr[:unique_edges, :]

        # Compute terminal positions
        src_idx = edge_index_unique[0, :]
        sink_idx = edge_index_unique[1, :]

        src_term_pos = positions[src_idx] + edge_attr_unique[:, :2]
        sink_term_pos = positions[sink_idx] + edge_attr_unique[:, 2:4]

        src_term_px = canvas_to_pixel(src_term_pos.numpy(), width, height)
        sink_term_px = canvas_to_pixel(sink_term_pos.numpy(), width, height)

        # Draw edges
        for i in range(unique_edges):
            draw.line(
                [tuple(src_term_px[i]), tuple(sink_term_px[i])],
                fill="gray",
                width=1
            )

        # Draw terminals
        for px in src_term_px:
            draw.ellipse([px[0]-3, px[1]-3, px[0]+3, px[1]+3], fill="black")
        for px in sink_term_px:
            draw.ellipse([px[0]-3, px[1]-3, px[0]+3, px[1]+3], fill="yellow")

    return np.array(image)


def load_chip_dataset(pickle_path):
    """Load chip placement dataset from pickle file"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_dataset_samples(dataset_path, num_samples=5, output_dir="chip_visualizations",
                              plot_edges=True, image_size=2048):
    """
    Visualize multiple samples from a chip placement dataset

    Args:
        dataset_path: path to dataset directory
        num_samples: number of samples to visualize
        output_dir: where to save visualization images
        plot_edges: whether to draw netlist connections
        image_size: output image size (square)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find pickle files
    dataset_path = Path(dataset_path)
    pickle_files = sorted(dataset_path.glob("*.pickle"))

    if not pickle_files:
        print(f"No pickle files found in {dataset_path}")
        return

    print(f"Found {len(pickle_files)} pickle files")

    # Load and visualize samples
    total_visualized = 0

    for pickle_file in pickle_files:
        if total_visualized >= num_samples:
            break

        print(f"Loading {pickle_file.name}...")
        sample_dict = load_chip_dataset(pickle_file)

        # Extract positions and data from dictionary
        positions = sample_dict['positions']
        data = sample_dict['H_graphs']

        print(f"  Visualizing sample {total_visualized + 1}/{num_samples}")

        # Convert positions if numpy
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions)

        # Create visualization
        img = visualize_chip_placement(positions, data,
                                     width=image_size,
                                     height=image_size,
                                     plot_edges=plot_edges)

        # Save image
        output_path = os.path.join(output_dir,
                                  f"chip_sample_{total_visualized:04d}.png")
        Image.fromarray(img).save(output_path, dpi=(300, 300))

        # Print statistics
        num_components = positions.shape[0]
        num_nets = data.edge_index.shape[1] // 2  # Undirected
        avg_size = data.x.mean(dim=0)
        density = sample_dict.get('densities', 'N/A')
        hpwl = sample_dict.get('Energies', 'N/A')

        print(f"    Components: {num_components}")
        print(f"    Nets: {num_nets}")
        print(f"    Avg size: ({avg_size[0]:.4f}, {avg_size[1]:.4f})")
        print(f"    Density: {density:.4f}" if isinstance(density, float) else f"    Density: {density}")
        print(f"    HPWL: {hpwl:.2f}" if isinstance(hpwl, (int, float)) else f"    HPWL: {hpwl}")
        print(f"    Saved: {output_path}")

        total_visualized += 1

    print(f"\nVisualization complete! Saved {total_visualized} images to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize chip placement datasets"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to chip placement dataset directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chip_visualizations",
        help="Output directory for images (default: chip_visualizations)"
    )
    parser.add_argument(
        "--no_edges",
        action="store_true",
        help="Do not plot netlist edges"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=2048,
        help="Output image size in pixels (default: 2048)"
    )

    args = parser.parse_args()

    visualize_dataset_samples(
        args.dataset_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        plot_edges=not args.no_edges,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main()
