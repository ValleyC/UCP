"""
Benchmark Data Utilities for UCP

Provides utilities to:
1. Parse raw benchmark files (DEF format, ISPD2005, IBM ICCAD04)
2. Cluster components using hMetis
3. Convert to UCP format (jraph.GraphsTuple)
4. Save/load placements
"""

import pickle
import numpy as np
import jraph
import torch
import re
import subprocess
import os
import tempfile
from pathlib import Path
from torch_geometric.data import Data
import sklearn.cluster


def load_graph_pickle(graph_path):
    """
    Load a preprocessed graph pickle file

    Args:
        graph_path: Path to graph*.pickle file

    Returns:
        torch_geometric.data.Data object with graph structure
    """
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    return data


def graph_to_jraph(data, chip_size=None):
    """
    Convert torch_geometric.Data to jraph.GraphsTuple

    Args:
        data: torch_geometric.data.Data with benchmark graph
        chip_size: Optional (width, height) tuple

    Returns:
        (H_graph, metadata) tuple
            H_graph: jraph.GraphsTuple
            metadata: dict with chip_size, is_ports, is_macros
    """
    # Extract data
    sizes = data.x.cpu().numpy() if torch.is_tensor(data.x) else data.x
    edge_index = data.edge_index.cpu().numpy() if torch.is_tensor(data.edge_index) else data.edge_index
    edge_attr = data.edge_attr.cpu().numpy() if torch.is_tensor(data.edge_attr) else data.edge_attr

    num_nodes = sizes.shape[0]
    num_edges = edge_index.shape[1]

    # Create jraph.GraphsTuple
    H_graph = jraph.GraphsTuple(
        nodes=sizes,  # (V, 2) component sizes
        edges=edge_attr,  # (E, 4) terminal offsets
        senders=edge_index[0, :],  # (E,) source node indices
        receivers=edge_index[1, :],  # (E,) target node indices
        n_node=np.array([num_nodes]),
        n_edge=np.array([num_edges]),
        globals=None
    )

    # Extract chip size
    if chip_size is None:
        if hasattr(data, 'chip_size'):
            if len(data.chip_size) == 4:  # [x_start, y_start, x_end, y_end]
                chip_size = (
                    float(data.chip_size[2] - data.chip_size[0]),
                    float(data.chip_size[3] - data.chip_size[1])
                )
            else:  # [width, height]
                chip_size = (float(data.chip_size[0]), float(data.chip_size[1]))
        else:
            chip_size = (2.0, 2.0)  # Default normalized canvas

    metadata = {
        'is_ports': data.is_ports.cpu().numpy() if hasattr(data, 'is_ports') else None,
        'is_macros': data.is_macros.cpu().numpy() if hasattr(data, 'is_macros') else None,
        'chip_size': chip_size,
    }

    return H_graph, metadata


def save_placement(positions, output_path):
    """
    Save placement as numpy array

    Args:
        positions: (V, 2) numpy array of component positions
        output_path: Path to save .pkl file
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)

    with open(output_path, 'wb') as f:
        pickle.dump(positions, f)

    print(f"Saved placement to {output_path}")


def load_placement(pkl_path):
    """
    Load placement from pickle file

    Args:
        pkl_path: Path to .pkl file

    Returns:
        (V, 2) numpy array of positions
    """
    with open(pkl_path, 'rb') as f:
        positions = pickle.load(f)
    return positions


def cluster_with_hmetis(graph_data, num_clusters=512, ubfactor=5, temp_dir=None, verbose=False):
    """
    Cluster graph components using hMetis hypergraph partitioning

    Args:
        graph_data: torch_geometric.data.Data object
        num_clusters: Number of clusters (default: 512)
        ubfactor: hMetis balance factor
        temp_dir: Temporary directory for hMetis files
        verbose: Print hMetis output

    Returns:
        cluster_assignments: (V,) numpy array of cluster IDs
        clustered_graph: New Data object with clustered nodes
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    # Build hyperedges from edge_index
    hyperedges = {}
    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None

    # Extract unique nets (each edge represents a net connection)
    _, E = edge_index.shape
    for i in range(E // 2):  # Only process forward edges (assuming bidirectional)
        u, v = edge_index[:, i].tolist()
        # Create hyperedge for this net
        net_id = i
        if net_id not in hyperedges:
            hyperedges[net_id] = set()
        hyperedges[net_id].add(u + 1)  # hMetis uses 1-indexed
        hyperedges[net_id].add(v + 1)

    # Write hMetis input file
    filename = os.path.join(temp_dir, 'edges.txt')
    num_vertices = graph_data.x.shape[0]

    with open(filename, 'w') as fp:
        fp.write(f'{len(hyperedges)} {num_vertices}\n')
        for net_id in sorted(hyperedges.keys()):
            vertices = sorted(hyperedges[net_id])
            fp.write(' '.join(map(str, vertices)) + '\n')

    # Run hMetis
    hmetis_path = './shmetis'  # Assumes shmetis is in PATH or current directory
    result = subprocess.run(
        [hmetis_path, filename, str(num_clusters), str(int(ubfactor))],
        capture_output=not verbose
    )

    # Parse output
    output_file = f'{filename}.part.{num_clusters}'
    with open(output_file, 'r') as fp:
        cluster_assignments = np.array([int(line.strip()) for line in fp.readlines()])

    return cluster_assignments


def create_clustered_graph(graph_data, cluster_assignments, num_clusters=512):
    """
    Create a new graph with clustered nodes

    Args:
        graph_data: Original Data object
        cluster_assignments: (V,) array of cluster IDs
        num_clusters: Number of clusters

    Returns:
        clustered_data: New Data object with num_clusters nodes
    """
    V = graph_data.x.shape[0]

    # Separate ports and macros (don't cluster these)
    is_ports = graph_data.is_ports if hasattr(graph_data, 'is_ports') else torch.zeros(V, dtype=torch.bool)
    is_macros = graph_data.is_macros if hasattr(graph_data, 'is_macros') else torch.zeros(V, dtype=torch.bool)

    # Create new node features (cluster sizes)
    cluster_sizes = []
    cluster_areas = torch.zeros(num_clusters)

    for i in range(V):
        size = graph_data.x[i]
        area = size[0] * size[1]

        if is_ports[i] or is_macros[i]:
            continue  # Skip ports and macros
        else:
            cluster_id = cluster_assignments[i]
            cluster_areas[cluster_id] += area

    # Create square clusters with aspect ratio 1:1
    for cluster_id in range(num_clusters):
        side_length = torch.sqrt(cluster_areas[cluster_id])
        cluster_sizes.append(torch.tensor([side_length, side_length]))

    # Add ports and macros
    for i in range(V):
        if is_ports[i] or is_macros[i]:
            cluster_sizes.append(graph_data.x[i])

    new_x = torch.stack(cluster_sizes)

    # Remap edges to clustered graph
    old_to_new_id = {}
    new_id = 0
    for i in range(V):
        if is_ports[i] or is_macros[i]:
            old_to_new_id[i] = num_clusters + (new_id - num_clusters)
            new_id += 1
        else:
            old_to_new_id[i] = cluster_assignments[i]

    # Update edge_index
    new_edge_index = []
    new_edge_attr = []
    _, E = graph_data.edge_index.shape

    for e in range(E):
        u, v = graph_data.edge_index[:, e].tolist()
        new_u = old_to_new_id[u]
        new_v = old_to_new_id[v]
        new_edge_index.append([new_u, new_v])
        if hasattr(graph_data, 'edge_attr'):
            new_edge_attr.append(graph_data.edge_attr[e])

    new_edge_index = torch.tensor(new_edge_index).T
    new_edge_attr = torch.stack(new_edge_attr) if new_edge_attr else None

    # Create new Data object
    clustered_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )

    # Update metadata
    new_is_ports = torch.zeros(new_x.shape[0], dtype=torch.bool)
    new_is_macros = torch.zeros(new_x.shape[0], dtype=torch.bool)

    for i in range(V):
        if is_ports[i]:
            new_is_ports[old_to_new_id[i]] = True
        if is_macros[i]:
            new_is_macros[old_to_new_id[i]] = True

    clustered_data.is_ports = new_is_ports
    clustered_data.is_macros = new_is_macros

    if hasattr(graph_data, 'chip_size'):
        clustered_data.chip_size = graph_data.chip_size

    return clustered_data


def load_benchmark_dataset(dataset_path, dataset_name="ispd2005"):
    """
    Load a benchmark dataset

    Args:
        dataset_path: Path to datasets directory
        dataset_name: Name of dataset (e.g., "ispd2005")

    Returns:
        List of (H_graph, metadata, file_idx) tuples
    """
    dataset_path = Path(dataset_path) / "graph" / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Find all graph*.pickle files
    graph_files = {
        int(re.search(r'\d+', p.name).group()): str(p)
        for p in dataset_path.rglob("graph*.pickle")
    }

    idx_list = sorted(graph_files.keys())

    benchmarks = []
    for idx in idx_list:
        graph_path = graph_files[idx]

        # Load graph
        data = load_graph_pickle(graph_path)

        # Convert to UCP format
        H_graph, metadata = graph_to_jraph(data)

        benchmarks.append((H_graph, metadata, idx))
        print(f"Loaded benchmark {idx}: {H_graph.nodes.shape[0]} components")

    return benchmarks


def normalize_positions(positions, chip_size=(2.0, 2.0), to_range=(-1, 1)):
    """
    Normalize positions to specified range

    Args:
        positions: (V, 2) positions
        chip_size: (width, height) of canvas
        to_range: Target range (default: [-1, 1])

    Returns:
        (V, 2) normalized positions
    """
    positions_norm = positions.copy()
    range_min, range_max = to_range

    # Scale to [0, 1] first
    positions_norm[:, 0] = positions[:, 0] / chip_size[0]
    positions_norm[:, 1] = positions[:, 1] / chip_size[1]

    # Scale to target range
    positions_norm = positions_norm * (range_max - range_min) + range_min

    return positions_norm


# Example usage
if __name__ == "__main__":
    print("Benchmark utilities loaded successfully")
