# import matplotlib.pyplot as plt
# from dolfin import *

# # Create two meshes
# # mesh1 = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), 10, 20)
# mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), 20, 10, "crossed")

# # Plot both meshes side by side
# fig, axs = plt.subplots(1, 1, figsize=(14, 6))

# plt.sca(axs)
# plot(mesh, title="RectangleMesh 10x20")
# axs.set_xlabel("x")
# axs.set_ylabel("y")
# axs.grid(True)


# plt.tight_layout()
# plt.show()

# V = VectorFunctionSpace(mesh, "P",2)

# # Define variational problem
# u = TrialFunction(V)
# v = TestFunction(V)

# E = 10.0
# nu = 0.3

# mu = E/2/(1+nu)
# lmda = E*nu/(1+nu)/(1-2*nu)

# def eps(v):
#     return sym(grad(v))

# def sigma(v):
#     return 2*mu*eps(v) + lmda*tr(eps(v))*Identity(len(v))


# zero = Constant((0.0),(0.0))

# def left_boundary(x, on_boundary):
#     return on_boundary and near(x[0], 0.0)


# bc = DirichletBC(V, zero, left_boundary)


# T = Constant((1.0, 0.0))


# # Define the bilinear form (stiffness matrix contribution)
# a = dot(grad(u), grad(v)) * dx

# boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# boundaries.set_all(0)



'''======================================================================================='''
from dolfin import *
import matplotlib.pyplot as plt

# Create mesh
mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), 10, 5)

# Mark boundary edges
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)

# Visualization
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))

# Use plot(mesh) instead of plot(mesh, ax=ax)
plot(mesh)

# Add the boundary labels
for facet in facets(mesh):
    if facet.exterior():
        midpoint = facet.midpoint()
        id = boundaries[facet.index()]
        plt.text(midpoint.x(), midpoint.y(), str(id), color='red', fontsize=8,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()








    