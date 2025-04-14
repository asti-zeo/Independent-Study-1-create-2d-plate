from dolfin import *
import matplotlib.pyplot as plt
from dolfin import Point
import numpy as np
import pandas as pd
import os

# Create a directory for output if it doesn't exist
output_dir = "simulation_results"
os.makedirs(output_dir, exist_ok=True)

shapes = ['Rectangle']
unit_size = 100
size_ratios = [[1,1]]  # Just test with a single size ratio
elements = [50]  # Use a very small mesh for testing
modulus_of_elasticity = [10000]

boundary_conditions = ['fixed']
boundary_location = ['left']

force_magnitudes = [1000]
force_locations = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [1.0, 1.0], [1.0, 0.5], [0.5, 1.0]]


# find the boundary/edges of the mesh

def on_boundary_left(x, on_boundary):
    return on_boundary and abs(x[0]) < 1e-14

def on_boundary_right(x, on_boundary):
    return on_boundary and abs(x[0] - width) < 1e-14

def on_boundary_top(x, on_boundary):
    return on_boundary and abs(x[1] - height) < 1e-14

def on_boundary_bottom(x, on_boundary):
    return on_boundary and abs(x[1]) < 1e-14

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*div(u)*Identity(len(u))  + 2.0*mu*epsilon(u)


# Loop through all parameters to generate diverse data
for shape in shapes:
    for size_ratio in size_ratios:
        for element in elements:
            for force_location in force_locations:
                for force_magnitude in force_magnitudes:
                    # Create a unique ID for this run
                    run_id = f"{shape}_ratio{size_ratio[0]}x{size_ratio[1]}_el{element}_loc{force_location[0]}x{force_location[1]}_mag{force_magnitude}"
                    print(f"\nProcessing configuration: {run_id}")
                    
                    E = 10.0e3
                    nu = 0.3

                    # Define the Lame' constants (common in linear elasticity)
                    mu    = E / (2.0*(1.0+nu))
                    lmbda = E*nu / ((1.0+nu)*(1.0-2.0*nu))

                    width = unit_size * size_ratio[0]
                    height = unit_size * size_ratio[1]

                    nx = int(element*size_ratio[0]/sum(size_ratio))
                    ny = int(element*size_ratio[1]/sum(size_ratio))

                    print(f"Creating mesh: {nx}x{ny} elements")
                    mesh = RectangleMesh(Point(0,0), Point(width, height), nx, ny)
                    mesh.init(1)

                    edge_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
                    edge_markers.set_all(0)

                    for edge in edges(mesh):
                        edge_markers[edge] = 1

                    edge_data = []
                    for edge in SubsetIterator(edge_markers, 1):
                        edge_data.append((edge.entities(0)[0], edge.entities(0)[1]))

                    df_edges = pd.DataFrame(edge_data, columns=['source', 'target'])
                    edges_file = os.path.join(output_dir, f"edges_{run_id}.csv")
                    df_edges.to_csv(edges_file, index=False)
                    print(f"Edge data saved to '{edges_file}'")

                    # Material properties
                    E = modulus_of_elasticity[0]  # Young's modulus
                    nu = 0.3  # Poisson's ratio

                    # Define the Lamé parameters
                    mu = E / (2.0 * (1.0 + nu))
                    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

                    V = VectorFunctionSpace(mesh, 'Lagrange', degree = 1)
                    u = TrialFunction(V)
                    v = TestFunction(V)
                    
                    # Define the bilinear form
                    a = inner(sigma(u), epsilon(v)) * dx
                    
                    # Initialize a zero load vector
                    L = dot(Constant((0.0, 0.0)), v) * dx
                    
                    fixed_displacement = Constant((0.0, 0.0))
                    
                    # apply boundary conditions
                    bc_left = DirichletBC(V, fixed_displacement, on_boundary_left)

                    # Assemble system
                    A = assemble(a)
                    b = assemble(L)

                    # Convert normalized force location to actual coordinates
                    force_x = width * force_location[0]
                    force_y = height * force_location[1]
                    
                    print(f"Applying point load at ({force_x}, {force_y}) with magnitude {force_magnitude}")
                    
                    # Find closest vertex to the desired force location
                    min_distance = float('inf')
                    closest_vertex = None
                    
                    for vertex in vertices(mesh):
                        dist = ((vertex.x(0) - force_x)**2 + (vertex.x(1) - force_y)**2)**0.5
                        if dist < min_distance:
                            min_distance = dist
                            closest_vertex = vertex
                    
                    if closest_vertex:
                        print(f"Closest vertex found at ({closest_vertex.x(0)}, {closest_vertex.x(1)}), distance: {min_distance}")
                        # Create point source at the closest vertex
                        point_source = PointSource(V.sub(1), Point(closest_vertex.x(0), closest_vertex.x(1)), -force_magnitude)
                        point_source.apply(b)
                    else:
                        print("Error: Could not find a suitable vertex for force application")

                    # Apply boundary conditions
                    [bc.apply(A, b) for bc in [bc_left]]
                    
                    # Create a function for the solution
                    u_sol = Function(V)
                    
                    # Solve the system
                    # Use a direct solver
                    solver = PETScLUSolver("mumps")
                    solver.solve(A, u_sol.vector(), b)

                    # save the solution
                    solution_file = os.path.join(output_dir, f"solution_{run_id}.xml")
                    file = File(solution_file)
                    file << u_sol
                    print(f"Solution saved to '{solution_file}'")
                    
                    # Save nodal displacements as CSV
                    node_coords = mesh.coordinates()
                    displacements = u_sol.compute_vertex_values(mesh)
                    num_nodes = len(node_coords)
                    
                    u_x = displacements[:num_nodes]
                    u_y = displacements[num_nodes:]
                    
                    # Create array for the results
                    displacement_data = np.column_stack((node_coords, u_x, u_y))
                    
                    # Save the data
                    displacement_file = os.path.join(output_dir, f"nodal_displacements_{run_id}.csv")
                    np.savetxt(displacement_file, displacement_data, 
                               delimiter=",", 
                               header="X,Y,UX,UY", 
                               comments="")
                    print(f"Nodal displacements saved to '{displacement_file}'")



                







