from dolfin import *
import numpy as np

# Create mesh
L = 1.0  # Length
W = 0.5  # Width
nx, ny = 50, 25  # Number of elements in each direction
domain = mesh.create_rectangle(MPI.COMM_WORLD, 
                              [np.array([0.0, 0.0]), np.array([L, W])],
                              [nx, ny],
                              mesh.CellType.triangle)

# Function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Material properties
E = 1.0e7  # Young's modulus
nu = 0.3   # Poisson's ratio
t = 0.01   # thickness
mu = E / (2.0 * (1.0 + nu))
lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Strain and stress
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return 2.0 * mu * epsilon(u) + lambda_ * tr(epsilon(u)) * Identity(2)

# Weak form
a = inner(sigma(u), epsilon(v))*dx
p = Constant(-100.0)
f = Constant((0, p*t))
L = dot(f, v)*dx

# Boundary conditions
def left(x, on_boundary):
    return on_boundary and near(x[0], 0)

def right(x, on_boundary):
    return on_boundary and near(x[0], L)

def top(x, on_boundary):
    return on_boundary and near(x[1], W)

def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)

# Apply BCs
bc_left = DirichletBC(V, Constant((0.0, 0.0)), left)
bc_right = DirichletBC(V.sub(1), Constant(0.0), right)
bc_top = DirichletBC(V.sub(1), Constant(0.0), top)
bc_bottom = DirichletBC(V.sub(1), Constant(0.0), bottom)

bcs = [bc_left, bc_right, bc_top, bc_bottom]

# Solve
u = Function(V)
solve(a == L, u, bcs)

# Post-process
stress = project(von_Mises(sigma(u)), FunctionSpace(mesh, 'P', 1))

# Save results
File('displacement.pvd') << u
File('stress.pvd') << stress

# Print max values
print(f"Maximum displacement: {np.max(np.abs(u.vector().get_local()))}")
print(f"Maximum von Mises stress: {np.max(stress.vector().get_local())}")





