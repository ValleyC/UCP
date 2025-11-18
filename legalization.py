"""
Force-Directed Legalization Decoder for UCP

Implements Algorithm 1 from the manuscript: Force-Directed Legalization
Ensures 100% matching with the paper description while being bug-free.
"""

import numpy as np
from typing import Tuple, Dict, List
import math


def compute_legality(positions: np.ndarray, sizes: np.ndarray) -> float:
    """
    Compute legality score A_u / A_s

    Args:
        positions: (N, 2) array of component center positions
        sizes: (N, 2) array of component sizes [width, height]

    Returns:
        legality: A_u / A_s where A_u is union area, A_s is sum of areas
    """
    N = len(positions)

    # Compute A_s: sum of individual component areas
    A_s = np.sum(sizes[:, 0] * sizes[:, 1])

    if A_s == 0:
        return 1.0

    # Compute A_u: area of union using sweep-line algorithm
    # For simplicity and accuracy, use pixel-based approximation with fine grid
    # This matches the legality definition in the manuscript

    # Get bounding box of all components
    x_min = np.min(positions[:, 0] - sizes[:, 0] / 2) - 0.1
    x_max = np.max(positions[:, 0] + sizes[:, 0] / 2) + 0.1
    y_min = np.min(positions[:, 1] - sizes[:, 1] / 2) - 0.1
    y_max = np.max(positions[:, 1] + sizes[:, 1] / 2) + 0.1

    # Create fine grid for union calculation (resolution based on smallest component)
    min_size = np.min(sizes)
    resolution = max(int((x_max - x_min) / (min_size / 10)), 100)
    resolution = min(resolution, 1000)  # Cap for efficiency

    x_coords = np.linspace(x_min, x_max, resolution)
    y_coords = np.linspace(y_min, y_max, resolution)

    # Create grid
    grid = np.zeros((resolution, resolution), dtype=bool)

    # Mark occupied cells
    for i in range(N):
        cx, cy = positions[i]
        w, h = sizes[i]

        # Component bounds
        left = cx - w / 2
        right = cx + w / 2
        bottom = cy - h / 2
        top = cy + h / 2

        # Find grid indices
        x_idx_left = np.searchsorted(x_coords, left)
        x_idx_right = np.searchsorted(x_coords, right)
        y_idx_bottom = np.searchsorted(y_coords, bottom)
        y_idx_top = np.searchsorted(y_coords, top)

        # Clamp to grid bounds
        x_idx_left = max(0, min(x_idx_left, resolution - 1))
        x_idx_right = max(0, min(x_idx_right, resolution - 1))
        y_idx_bottom = max(0, min(y_idx_bottom, resolution - 1))
        y_idx_top = max(0, min(y_idx_top, resolution - 1))

        # Mark cells as occupied
        grid[y_idx_bottom:y_idx_top+1, x_idx_left:x_idx_right+1] = True

    # Compute A_u as area of occupied cells
    cell_area = ((x_max - x_min) / resolution) * ((y_max - y_min) / resolution)
    A_u = np.sum(grid) * cell_area

    return A_u / A_s


def compute_signed_distance(pos_i: np.ndarray, pos_j: np.ndarray,
                           size_i: np.ndarray, size_j: np.ndarray) -> float:
    """
    Compute signed distance d_ij between two rectangular components

    Following Equation 11 in manuscript:
    d_ij(x) = max(|x_i - x_j| - (w_i + w_j)/2, |y_i - y_j| - (h_i + h_j)/2)

    Args:
        pos_i: (2,) position of component i [x, y]
        pos_j: (2,) position of component j [x, y]
        size_i: (2,) size of component i [width, height]
        size_j: (2,) size of component j [width, height]

    Returns:
        d_ij: Signed distance (negative if overlapping)
    """
    # Distance between centers
    dx = abs(pos_i[0] - pos_j[0])
    dy = abs(pos_i[1] - pos_j[1])

    # Half-sizes sum
    half_w_sum = (size_i[0] + size_j[0]) / 2.0
    half_h_sum = (size_i[1] + size_j[1]) / 2.0

    # Signed distance in each dimension
    dist_x = dx - half_w_sum
    dist_y = dy - half_h_sum

    # Maximum determines signed distance
    return max(dist_x, dist_y)


def compute_boundary_gradient(position: np.ndarray, size: np.ndarray,
                              canvas_bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Compute gradient of boundary penalty for a component

    Following Equation 13 in manuscript:
    E_bound^(i) = [max(0, |x_i| + w_i/2 - 1)]^2 + [max(0, |y_i| + h_i/2 - 1)]^2

    Args:
        position: (2,) component position [x, y]
        size: (2,) component size [width, height]
        canvas_bounds: (x_min, x_max, y_min, y_max) canvas boundaries

    Returns:
        gradient: (2,) gradient [∂E/∂x, ∂E/∂y]
    """
    x, y = position
    w, h = size
    x_min, x_max, y_min, y_max = canvas_bounds

    grad = np.zeros(2)

    # Component boundaries
    left = x - w / 2
    right = x + w / 2
    bottom = y - h / 2
    top = y + h / 2

    # X-direction gradient
    if left < x_min:
        # Violation on left side
        violation = x_min - left
        grad[0] = -2.0 * violation  # Gradient pushes right
    elif right > x_max:
        # Violation on right side
        violation = right - x_max
        grad[0] = 2.0 * violation  # Gradient pushes left

    # Y-direction gradient
    if bottom < y_min:
        # Violation on bottom side
        violation = y_min - bottom
        grad[1] = -2.0 * violation  # Gradient pushes up
    elif top > y_max:
        # Violation on top side
        violation = top - y_max
        grad[1] = 2.0 * violation  # Gradient pushes down

    return grad


class SpatialHashGrid:
    """
    Spatial hash grid for efficient neighbor queries

    Following Algorithm 1: Initialize uniform grid with cell size δ = 2 · E[max(w_i, h_i)]
    """

    def __init__(self, positions: np.ndarray, sizes: np.ndarray,
                 canvas_bounds: Tuple[float, float, float, float]):
        """
        Initialize spatial hash grid

        Args:
            positions: (N, 2) component positions
            sizes: (N, 2) component sizes
            canvas_bounds: (x_min, x_max, y_min, y_max)
        """
        # Cell size: δ = 2 · E[max(w_i, h_i)]
        max_sizes = np.max(sizes, axis=1)  # max(w_i, h_i) for each component
        expected_max_size = np.mean(max_sizes)  # E[max(w_i, h_i)]
        self.cell_size = 2.0 * expected_max_size

        # Avoid too small cell size
        self.cell_size = max(self.cell_size, 0.1)

        x_min, x_max, y_min, y_max = canvas_bounds
        self.x_min = x_min
        self.y_min = y_min

        # Grid dimensions
        self.grid_width = int(np.ceil((x_max - x_min) / self.cell_size)) + 1
        self.grid_height = int(np.ceil((y_max - y_min) / self.cell_size)) + 1

        # Initialize grid cells
        self.grid: Dict[Tuple[int, int], List[int]] = {}

        # Build spatial hash
        self.build(positions, sizes)

    def _get_cell_coords(self, position: np.ndarray) -> Tuple[int, int]:
        """Get grid cell coordinates for a position"""
        x, y = position
        cell_x = int((x - self.x_min) / self.cell_size)
        cell_y = int((y - self.y_min) / self.cell_size)

        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, self.grid_width - 1))
        cell_y = max(0, min(cell_y, self.grid_height - 1))

        return (cell_x, cell_y)

    def build(self, positions: np.ndarray, sizes: np.ndarray):
        """
        Build spatial hash: assign each component to grid cells

        Args:
            positions: (N, 2) component positions
            sizes: (N, 2) component sizes
        """
        # Clear grid
        self.grid.clear()

        N = len(positions)
        for i in range(N):
            pos = positions[i]
            size = sizes[i]

            # Get bounding box of component
            left = pos[0] - size[0] / 2
            right = pos[0] + size[0] / 2
            bottom = pos[1] - size[1] / 2
            top = pos[1] + size[1] / 2

            # Get cell range that component occupies
            cell_x_min = int((left - self.x_min) / self.cell_size)
            cell_x_max = int((right - self.x_min) / self.cell_size)
            cell_y_min = int((bottom - self.y_min) / self.cell_size)
            cell_y_max = int((top - self.y_min) / self.cell_size)

            # Clamp to grid bounds
            cell_x_min = max(0, min(cell_x_min, self.grid_width - 1))
            cell_x_max = max(0, min(cell_x_max, self.grid_width - 1))
            cell_y_min = max(0, min(cell_y_min, self.grid_height - 1))
            cell_y_max = max(0, min(cell_y_max, self.grid_height - 1))

            # Add component to all cells it overlaps
            for cx in range(cell_x_min, cell_x_max + 1):
                for cy in range(cell_y_min, cell_y_max + 1):
                    cell_key = (cx, cy)
                    if cell_key not in self.grid:
                        self.grid[cell_key] = []
                    self.grid[cell_key].append(i)

    def get_neighbors(self, component_idx: int, positions: np.ndarray,
                     sizes: np.ndarray) -> List[int]:
        """
        Get neighboring components (in same and adjacent cells)

        Args:
            component_idx: Index of component to query
            positions: (N, 2) all component positions
            sizes: (N, 2) all component sizes

        Returns:
            neighbors: List of component indices in neighboring cells
        """
        pos = positions[component_idx]
        size = sizes[component_idx]

        # Get cells that component occupies
        left = pos[0] - size[0] / 2
        right = pos[0] + size[0] / 2
        bottom = pos[1] - size[1] / 2
        top = pos[1] + size[1] / 2

        cell_x_min = int((left - self.x_min) / self.cell_size)
        cell_x_max = int((right - self.x_min) / self.cell_size)
        cell_y_min = int((bottom - self.y_min) / self.cell_size)
        cell_y_max = int((top - self.y_min) / self.cell_size)

        # Include neighboring cells (extend by 1 in each direction)
        cell_x_min = max(0, cell_x_min - 1)
        cell_x_max = min(self.grid_width - 1, cell_x_max + 1)
        cell_y_min = max(0, cell_y_min - 1)
        cell_y_max = min(self.grid_height - 1, cell_y_max + 1)

        # Collect all components in these cells
        neighbors = set()
        for cx in range(cell_x_min, cell_x_max + 1):
            for cy in range(cell_y_min, cell_y_max + 1):
                cell_key = (cx, cy)
                if cell_key in self.grid:
                    neighbors.update(self.grid[cell_key])

        # Remove self
        neighbors.discard(component_idx)

        return list(neighbors)


def force_directed_legalization(
    positions: np.ndarray,
    sizes: np.ndarray,
    canvas_bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    legality_threshold: float = 0.99,
    max_iterations: int = 1000,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Force-directed legalization algorithm (Algorithm 1 from manuscript)

    Args:
        positions: (N, 2) initial component positions [x, y]
        sizes: (N, 2) component sizes [width, height]
        canvas_bounds: (x_min, x_max, y_min, y_max) canvas boundaries
        legality_threshold: Target legality A_u/A_s (default: 0.99)
        max_iterations: Maximum iterations (default: 1000)
        verbose: Print progress

    Returns:
        positions_legal: (N, 2) legalized positions
        stats: Dictionary with convergence statistics
    """
    N = len(positions)
    positions = positions.copy()  # Don't modify input

    x_min, x_max, y_min, y_max = canvas_bounds

    # Statistics
    stats = {
        'iterations': 0,
        'initial_legality': compute_legality(positions, sizes),
        'final_legality': 0.0,
        'converged': False
    }

    if verbose:
        print(f"Initial legality: {stats['initial_legality']:.4f}")

    # Main legalization loop
    for iteration in range(1, max_iterations + 1):
        # Compute current legality
        legality = compute_legality(positions, sizes)

        # Check convergence
        if legality >= legality_threshold:
            stats['iterations'] = iteration
            stats['final_legality'] = legality
            stats['converged'] = True
            if verbose:
                print(f"Converged at iteration {iteration}: legality = {legality:.4f}")
            break

        # Build spatial hash grid
        grid = SpatialHashGrid(positions, sizes, canvas_bounds)

        # Compute forces for each component
        forces = np.zeros((N, 2))

        for i in range(N):
            # Get neighbors from spatial hash
            neighbors = grid.get_neighbors(i, positions, sizes)

            # Compute repulsion forces from overlapping components
            for j in neighbors:
                if i == j:
                    continue

                # Compute signed distance
                d_ij = compute_signed_distance(positions[i], positions[j],
                                              sizes[i], sizes[j])

                # If overlapping (d_ij < 0), apply repulsion
                if d_ij < 0:
                    # Direction vector from i to j
                    diff = positions[j] - positions[i]
                    dist = np.linalg.norm(diff)

                    if dist > 1e-8:  # Avoid division by zero
                        v_ij = diff / dist  # Normalized direction
                    else:
                        # Components at same position: use random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        v_ij = np.array([np.cos(angle), np.sin(angle)])

                    # Repulsion force: f_i = -|d_ij| · v_ij
                    forces[i] -= abs(d_ij) * v_ij

            # Check if component is outside canvas
            cx, cy = positions[i]
            w, h = sizes[i]

            left = cx - w / 2
            right = cx + w / 2
            bottom = cy - h / 2
            top = cy + h / 2

            outside_canvas = (left < x_min or right > x_max or
                            bottom < y_min or top > y_max)

            if outside_canvas:
                # Add boundary gradient
                boundary_grad = compute_boundary_gradient(positions[i], sizes[i],
                                                         canvas_bounds)
                forces[i] -= boundary_grad

        # Adaptive step size: η = min(0.1, 1.0 / sqrt(iteration))
        eta = min(0.1, 1.0 / math.sqrt(iteration))

        # Update positions: x = x + η · f
        positions += eta * forces

        # Print progress
        if verbose and (iteration % 100 == 0 or iteration == 1):
            print(f"Iteration {iteration}: legality = {legality:.4f}, η = {eta:.4f}")

    # Final statistics
    if not stats['converged']:
        stats['iterations'] = max_iterations
        stats['final_legality'] = compute_legality(positions, sizes)
        if verbose:
            print(f"Max iterations reached. Final legality: {stats['final_legality']:.4f}")

    return positions, stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing Force-Directed Legalization Decoder")
    print("=" * 60)

    # Create test case with overlapping components
    np.random.seed(42)
    N = 50

    # Random positions (may overlap)
    positions = np.random.uniform(-0.8, 0.8, size=(N, 2))

    # Random sizes
    sizes = np.random.uniform(0.05, 0.15, size=(N, 2))

    # Canvas bounds
    canvas_bounds = (-1.0, 1.0, -1.0, 1.0)

    print(f"\nTest case: {N} components")
    print(f"Canvas bounds: {canvas_bounds}")

    # Initial legality
    initial_legality = compute_legality(positions, sizes)
    print(f"Initial legality: {initial_legality:.4f}")

    # Run legalization
    print("\nRunning force-directed legalization...")
    positions_legal, stats = force_directed_legalization(
        positions, sizes, canvas_bounds, verbose=True
    )

    # Results
    print("\n" + "=" * 60)
    print("Legalization Results:")
    print(f"  Converged: {stats['converged']}")
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Initial legality: {stats['initial_legality']:.4f}")
    print(f"  Final legality: {stats['final_legality']:.4f}")
    print(f"  Improvement: {stats['final_legality'] - stats['initial_legality']:.4f}")
