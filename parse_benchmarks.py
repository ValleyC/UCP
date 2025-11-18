"""
Raw Benchmark Parser for UCP

Parses raw DEF/LEF files from ISPD2005 and IBM ICCAD04 benchmarks
and converts them to preprocessed graph format.

Supports:
- DEF file parsing
- Component clustering with hMetis
- Macro-only extraction
- Graph generation for UCP

Usage:
    python parse_benchmarks.py \\
        --input_dir benchmarks/ispd2005_raw \\
        --output_dir datasets/graph/ispd2005 \\
        --mode clustered \\
        --num_clusters 512
"""

import argparse
import re
import os
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import subprocess
import tempfile


class DEFParser:
    """Parse DEF (Design Exchange Format) files"""

    def __init__(self, def_path):
        self.def_path = def_path
        self.components = []
        self.nets = []
        self.chip_size = None

        self._parse()

    def _parse(self):
        """Parse DEF file"""
        with open(self.def_path, 'r') as f:
            content = f.read()

        # Parse DIEAREA
        diearea_match = re.search(r'DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)', content)
        if diearea_match:
            x1, y1, x2, y2 = map(int, diearea_match.groups())
            self.chip_size = (x2 - x1, y2 - y1)

        # Parse COMPONENTS section
        components_section = self._extract_section(content, 'COMPONENTS')
        if components_section:
            self._parse_components(components_section)

        # Parse NETS section
        nets_section = self._extract_section(content, 'NETS')
        if nets_section:
            self._parse_nets(nets_section)

    def _extract_section(self, content, section_name):
        """Extract a section from DEF file"""
        pattern = f'{section_name}.*?END {section_name}'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(0) if match else None

    def _parse_components(self, section):
        """Parse COMPONENTS section"""
        lines = section.split('\n')

        for line in lines[1:-1]:  # Skip first and last line
            line = line.strip()
            if line.startswith('-'):
                # Parse component
                parts = line.split()
                name = parts[1]

                # Extract placement info
                placement_idx = line.find('+ PLACED')
                if placement_idx != -1:
                    placement_str = line[placement_idx:]
                    placement_parts = placement_str.split()
                    x = int(placement_parts[2])
                    y = int(placement_parts[3])
                else:
                    x, y = 0, 0

                # For now, assume unit size (will be updated with LEF info)
                width, height = 1, 1

                self.components.append({
                    'name': name,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                })

    def _parse_nets(self, section):
        """Parse NETS section"""
        lines = section.split('\n')

        current_net = None
        for line in lines[1:-1]:
            line = line.strip()
            if line.startswith('-'):
                # New net
                if current_net:
                    self.nets.append(current_net)

                parts = line.split()
                net_name = parts[1]
                current_net = {
                    'name': net_name,
                    'connections': []
                }

            elif line.startswith('('):
                # Connection
                match = re.search(r'\(\s*(\S+)\s+(\S+)\s*\)', line)
                if match:
                    comp_name = match.group(1)
                    pin_name = match.group(2)
                    current_net['connections'].append({
                        'component': comp_name,
                        'pin': pin_name
                    })

        if current_net:
            self.nets.append(current_net)

    def to_graph(self):
        """Convert to torch_geometric.Data"""
        # Create component index
        comp_name_to_idx = {c['name']: i for i, c in enumerate(self.components)}

        # Build node features (sizes)
        sizes = []
        for comp in self.components:
            sizes.append([comp['width'], comp['height']])
        x = torch.tensor(sizes, dtype=torch.float32)

        # Build edges from nets
        edge_list = []
        edge_attrs = []

        for net in self.nets:
            connections = net['connections']
            if len(connections) < 2:
                continue

            # Create edges between all pairs (star topology)
            for i in range(len(connections)):
                for j in range(i + 1, len(connections)):
                    comp_i = connections[i]['component']
                    comp_j = connections[j]['component']

                    if comp_i in comp_name_to_idx and comp_j in comp_name_to_idx:
                        idx_i = comp_name_to_idx[comp_i]
                        idx_j = comp_name_to_idx[comp_j]

                        # Add bidirectional edges
                        edge_list.append([idx_i, idx_j])
                        edge_list.append([idx_j, idx_i])

                        # Terminal offsets (simplified - use component centers)
                        edge_attrs.append([0, 0, 0, 0])
                        edge_attrs.append([0, 0, 0, 0])

        edge_index = torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.zeros((0, 4), dtype=torch.float32)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Add metadata
        data.chip_size = torch.tensor(self.chip_size) if self.chip_size else torch.tensor([1000, 1000])
        data.is_ports = torch.zeros(len(self.components), dtype=torch.bool)
        data.is_macros = torch.zeros(len(self.components), dtype=torch.bool)

        return data


def cluster_graph(graph_data, num_clusters=512, hmetis_path='./shmetis'):
    """
    Cluster graph using hMetis

    Args:
        graph_data: torch_geometric.data.Data
        num_clusters: Number of clusters
        hmetis_path: Path to hMetis binary

    Returns:
        clustered_data: Data object with clustered nodes
    """
    from benchmark_utils import cluster_with_hmetis, create_clustered_graph

    # Run hMetis clustering
    cluster_assignments = cluster_with_hmetis(
        graph_data,
        num_clusters=num_clusters,
        ubfactor=5,
        verbose=False
    )

    # Create clustered graph
    clustered_data = create_clustered_graph(
        graph_data,
        cluster_assignments,
        num_clusters=num_clusters
    )

    return clustered_data


def extract_macros_only(graph_data):
    """
    Extract macro-only graph

    Args:
        graph_data: Data with is_macros attribute

    Returns:
        macro_data: Data with macros only
    """
    is_macros = graph_data.is_macros

    # Get macro indices
    macro_indices = torch.where(is_macros)[0]

    if len(macro_indices) == 0:
        # No macros - return as is
        return graph_data

    # Extract macro nodes
    new_x = graph_data.x[macro_indices]

    # Remap edges
    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(macro_indices)}

    new_edge_list = []
    new_edge_attrs = []

    for e in range(graph_data.edge_index.shape[1]):
        u, v = graph_data.edge_index[:, e].tolist()
        if u in old_to_new and v in old_to_new:
            new_edge_list.append([old_to_new[u], old_to_new[v]])
            if graph_data.edge_attr is not None:
                new_edge_attrs.append(graph_data.edge_attr[e])

    new_edge_index = torch.tensor(new_edge_list, dtype=torch.long).T if new_edge_list else torch.zeros((2, 0), dtype=torch.long)
    new_edge_attr = torch.stack(new_edge_attrs) if new_edge_attrs else None

    # Create new Data
    macro_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
    )

    macro_data.chip_size = graph_data.chip_size
    macro_data.is_ports = torch.zeros(len(new_x), dtype=torch.bool)
    macro_data.is_macros = torch.ones(len(new_x), dtype=torch.bool)

    return macro_data


def main():
    parser = argparse.ArgumentParser(description='Parse raw benchmarks to graph format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with raw DEF files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for graph*.pickle files')
    parser.add_argument('--mode', type=str, default='clustered',
                        choices=['raw', 'clustered', 'macro-only'],
                        help='Processing mode')
    parser.add_argument('--num_clusters', type=int, default=512,
                        help='Number of clusters (for clustered mode)')
    parser.add_argument('--hmetis_path', type=str, default='./shmetis',
                        help='Path to hMetis binary')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Benchmark Parser")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing mode: {args.mode}")
    if args.mode == 'clustered':
        print(f"Number of clusters: {args.num_clusters}")
    print("=" * 80)

    # Find all DEF files
    def_files = sorted(list(input_dir.glob("*.def")))

    if len(def_files) == 0:
        print(f"ERROR: No DEF files found in {input_dir}")
        return

    print(f"\nFound {len(def_files)} DEF files")

    # Process each file
    for idx, def_file in enumerate(def_files):
        print(f"\nProcessing {def_file.name}...")

        # Parse DEF
        parser = DEFParser(def_file)
        graph_data = parser.to_graph()

        print(f"  Components: {graph_data.x.shape[0]}")
        print(f"  Nets: {graph_data.edge_index.shape[1] // 2}")

        # Apply mode-specific processing
        if args.mode == 'clustered':
            print(f"  Clustering to {args.num_clusters} clusters...")
            graph_data = cluster_graph(graph_data, args.num_clusters, args.hmetis_path)
            print(f"  After clustering: {graph_data.x.shape[0]} nodes")

        elif args.mode == 'macro-only':
            print(f"  Extracting macros only...")
            graph_data = extract_macros_only(graph_data)
            print(f"  Macros: {graph_data.x.shape[0]}")

        # Save
        output_path = output_dir / f"graph{idx}.pickle"
        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)

        print(f"  Saved to {output_path}")

    # Create config file
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        f.write(f"train_samples: {int(len(def_files) * 0.7)}\n")
        f.write(f"val_samples: {int(len(def_files) * 0.3)}\n")
        f.write(f"scale: 1.0\n")
        f.write(f"chip_width: 2.0\n")
        f.write(f"chip_height: 2.0\n")

    print(f"\nConfig saved to {config_path}")
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
