from dolfin import *
import matplotlib.pyplot as plt
from dolfin import Point
import numpy as np
import pandas as pd



def boundary_left(x, on_boundary):
    return on_boundary and abs(x[0]) < 1e-14

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*div(u)*Identity(len(u))  + 2.0*mu*epsilon(u)


class PointLoad(UserExpression):
    def __init__(self, x_target, y_target,force_magnitude, eps=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.x_target = x_target
        self.y_target = y_target
        self.force_magnitude = force_magnitude
        self.eps = eps
        self.loaded_points = []



    def eval(self, values, x):
        if abs(x[0]-self.x_target) < self.eps and abs(x[1] -  self.y_target) < self.eps:
            values[0] = 0.0
            values[1] = self.force_magnitude
            self.loaded_points.append([x[0], x[1]])

        else:
            values[0] = 0.0
            values[1] = 0.0
            print("No point found")
    def value_shape(self):
        """Ensure the function is vector-valued."""
        return (2,)  # 2D vector (x, y)

        
    


# Plate dimensions
L = 1.0  # length in x-direction
H = 0.2  # height in y-direction

# Number of elements in each direction
nx = 20
ny = 5

# Create rectangular mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)
mesh.init(1)

# Create a MeshFunction to mark the edges
edge_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
edge_markers.set_all(0)

# Iterate over the edges and mark them
for edge in edges(mesh):
    edge_markers[edge] = 1

# Extract edge data
edge_data = []
for edge in SubsetIterator(edge_markers, 1):
    edge_data.append((edge.entities(0)[0], edge.entities(0)[1]))

# Convert to DataFrame and save
df_edges = pd.DataFrame(edge_data, columns=['source', 'target'])
df_edges.to_csv('edges.csv', index=False)
print("Edge data saved to 'edges.csv'")


# define the function space
V = VectorFunctionSpace(mesh, 'Lagrange', degree = 1)

# plot(mesh)
# plt.title("2D Plate Mesh")
# plt.show()

# set the displacement to zero at left side of the plate
fixed_displacement = Constant((0.0, 0.0))

# set the solution to zero since its fixed
bc_left = DirichletBC(V, fixed_displacement, boundary_left)

bcs = [bc_left]

#material properties
# youngs modulus
E = 10.0e3
nu = 0.3

# Define the Lame' constants (common in linear elasticity)
mu    = E / (2.0*(1.0+nu))
lmbda = E*nu / ((1.0+nu)*(1.0-2.0*nu))

# Find center point
center_point = Point(L/2, H/2)

# Create point source at center
force_magnitude = -1000.0  # Negative for downward force


u = TrialFunction(V)
v = TestFunction(V)

# f = Constant((1000, 0.0))
# F = assemble(action(inner(sigma(u), epsilon(v))*dx, u))
# point_source = PointSource(V.sub(1), center_point, force_magnitude)
# point_source.apply(F)

f = PointLoad(L, H, -1000000,eps = 0.02, degree = 2)


a = inner(sigma(u),epsilon(v))*dx

L = dot(f,v)*dx

u_sol = Function(V)
solve( a == L, u_sol, bcs)

print(mesh.coordinates())

# 7. Plot Mesh & Displacement
# plot(mesh, title="Mesh")
# plt.show()

# plot(u_sol, mode="displacement", title="Displacement")
# plt.show()
f_points = np.array(f.loaded_points)
plt.figure()

p = plot(u_sol, mode="displacement", title="Displacement")
plt.plot(f_points[:,0], f_points[:,1], 'ro', label='Loaded Nodes', markersize=2)
plot(mesh, color='black', linewidth=0.2, alpha=0.7)
plt.colorbar(p, label='Displacement magnitude')
plt.legend()
plt.savefig('displacement_plot.png', dpi=300, bbox_inches='tight')  # Save before showing
plt.show()



# if len(f_points) > 0:
#     plt.figure()
#     plot(mesh, title="Mesh with Load Points")
#     plt.plot(f_points[:,0], f_points[:,1], 'ro', label='Loaded Nodes', markersize=10)
#     plt.legend()
#     plt.show()





node_coords = mesh.coordinates()
displacements = u_sol.compute_vertex_values(mesh)
num_nodes = len(node_coords)

u_x = displacements[:num_nodes]
u_y = displacements[num_nodes:]
# Create array of load indicators (1 where load applied, 0 otherwise)
load_indicators = np.zeros(len(node_coords))
for point in f_points:
    # Find nodes where load is applied (within small tolerance due to floating point)
    matches = np.all(np.abs(node_coords - point) < 1e-10, axis=1)
    load_indicators[matches] = 1

# Determine boundary nodes
boundary_nodes = np.zeros(num_nodes, dtype=int)
for i, coord in enumerate(node_coords):
    if boundary_left(coord, True):
        boundary_nodes[i] = 1

# Stack coordinates, displacements, load indicators, and boundary information
displacement_data = np.column_stack((node_coords, u_x, u_y, load_indicators, boundary_nodes))

# Save the data with an additional column for boundary information
np.savetxt("nodal_displacements.csv", displacement_data, delimiter=",", header="X,Y,UX,UY,LOAD,BOUNDARY", comments="")


