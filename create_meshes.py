import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import Delaunay

def create_2d_plate_mesh(length, height, nx, ny, random_points=False):
    """
    Create a 2D triangular mesh for a rectangular plate using Delaunay triangulation.
    
    Parameters:
    length: float - length of the plate (x-direction)
    height: float - height of the plate (y-direction)
    nx: int - approximate number of points in x-direction
    ny: int - approximate number of points in y-direction
    random_points: bool - if True, adds some randomness to point positions
    
    Returns:
    nodes: array of node coordinates
    elements: array of element connectivity
    """
    if random_points:
        # Create slightly randomized points
        x = np.linspace(0, length, nx)
        y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(x, y)
        
        # Add some random perturbation to interior points
        dx = length / (nx - 1) * 0.3  # 30% of grid spacing
        dy = height / (ny - 1) * 0.3
        
        # Only perturb interior points
        mask = ((X > 0) & (X < length) & (Y > 0) & (Y < height))
        X[mask] += np.random.uniform(-dx, dx, size=X[mask].shape)
        Y[mask] += np.random.uniform(-dy, dy, size=Y[mask].shape)
        
        # Add boundary points to ensure proper domain coverage
        boundary_x = np.concatenate([
            np.linspace(0, length, nx),  # bottom
            np.linspace(0, length, nx),  # top
            np.zeros(ny),                # left
            np.full(ny, length)          # right
        ])
        boundary_y = np.concatenate([
            np.zeros(nx),                # bottom
            np.full(nx, height),         # top
            np.linspace(0, height, ny),  # left
            np.linspace(0, height, ny)   # right
        ])
        
        # Combine interior and boundary points
        points_x = np.concatenate([X.flatten(), boundary_x])
        points_y = np.concatenate([Y.flatten(), boundary_y])
        
    else:
        # Create regular grid points
        x = np.linspace(0, length, nx)
        y = np.linspace(0, height, ny)
        X, Y = np.meshgrid(x, y)
        points_x = X.flatten()
        points_y = Y.flatten()
    
    # Combine coordinates into points array
    points = np.column_stack((points_x, points_y))
    
    # Remove duplicate points that might have been created
    points = np.unique(points, axis=0)
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    return points, tri.simplices

def create_graded_mesh(length, height, nx, ny, bias=2.0):
    """
    Create a 2D triangular mesh with elements increasing in size from center.
    
    Parameters:
    length: float - length of the plate (x-direction)
    height: float - height of the plate (y-direction)
    nx, ny: int - number of points in each direction
    bias: float - controls mesh size growth rate (higher = faster growth)
    """
    # Create points with increasing spacing from center
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    
    # Apply sinh transformation to create bias towards center
    x = length/2 * (1 + np.sinh(bias * x)/np.sinh(bias))
    y = height/2 * (1 + np.sinh(bias * y)/np.sinh(bias))
    
    # Create grid of points
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    return points, tri.simplices

def create_directional_graded_mesh(length, height, nx, ny, x_bias=2.0, y_bias=2.0):
    """
    Create a 2D triangular mesh with elements increasing in size along x and y directions.
    
    Parameters:
    length: float - length of the plate (x-direction)
    height: float - height of the plate (y-direction)
    nx, ny: int - number of points in each direction
    x_bias: float - controls mesh size growth rate in x direction
    y_bias: float - controls mesh size growth rate in y direction
    
    Returns:
    nodes: array of node coordinates
    elements: array of element connectivity
    """
    # Create points with increasing spacing from left to right
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    
    # Apply exponential transformation to create bias
    x = length * (np.exp(x_bias * x) - 1) / (np.exp(x_bias) - 1)
    y = height * (np.exp(y_bias * y) - 1) / (np.exp(y_bias) - 1)
    
    # Create grid of points
    X, Y = np.meshgrid(x, y)
    points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    return points, tri.simplices

def create_sectioned_graded_mesh(length, height, nx, ny, y_sections, x_bias=2.0, y_bias=2.0):
    """
    Create a 2D triangular mesh with sections at specified y-distances and continuously increasing element sizes.
    """
    all_points = []
    
    # Create points with increasing spacing from left to right
    x = np.linspace(0, 1, nx)
    x = length * (np.exp(x_bias * x) - 1) / (np.exp(x_bias) - 1)
    
    # Create a single continuous y-coordinate array
    total_ny = ny * (len(y_sections) - 1)
    y_continuous = np.linspace(0, 1, total_ny)
    y_continuous = height * (np.exp(y_bias * y_continuous) - 1) / (np.exp(y_bias) - 1)
    
    # Create mesh for each section
    for i in range(len(y_sections) - 1):
        section_start = y_sections[i]
        section_end = y_sections[i+1]
        
        # Select points within this section
        y_mask = (y_continuous >= section_start) & (y_continuous <= section_end)
        y_section = y_continuous[y_mask]
        
        # Ensure section boundaries are included
        if y_section[0] != section_start:
            y_section = np.insert(y_section, 0, section_start)
        if y_section[-1] != section_end:
            y_section = np.append(y_section, section_end)
        
        # Create grid for this section
        X, Y = np.meshgrid(x, y_section)
        section_points = np.column_stack((X.flatten(), Y.flatten()))
        all_points.append(section_points)
    
    # Combine all points
    points = np.vstack(all_points)
    
    # Remove duplicate points that might occur at section boundaries
    points = np.unique(points, axis=0)
    
    # Create Delaunay triangulation
    tri = Delaunay(points)
    
    return points, tri.simplices

def plot_mesh(nodes, elements):
    """
    Plot the triangular mesh.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot triangles
    for element in elements:
        triangle = np.vstack((nodes[element], nodes[element[0]]))
        plt.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=0.5)
    
    # Plot nodes
    plt.plot(nodes[:, 0], nodes[:, 1], 'r.', markersize=3)
    
    plt.axis('equal')
    plt.grid(False)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graded Triangular Mesh')
    plt.show()

if __name__ == "__main__":
    # Example usage
    length = 10.0
    height = 10.0
    nx = 30
    ny = 10  # points per section
    
    # Define sections at y = 0, 2, 5, and 8
    y_sections = [0, 2, 5, 8]
    
    # Create sectioned graded mesh with continuous size increase
    print("\nCreating sectioned graded mesh")
    nodes, elements = create_sectioned_graded_mesh(
        length, height, nx, ny, 
        y_sections=y_sections,
        x_bias=3.5, y_bias=3.0  # Adjusted for better progression
    )
    plot_mesh(nodes, elements)
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of elements: {len(elements)}")
