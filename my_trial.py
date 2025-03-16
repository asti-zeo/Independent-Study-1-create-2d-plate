from dolfin import *


mesh =  UnitSquareMesh(10,10)

element = FiniteElement('Lagrange', mesh.ufl_cell(), 1)

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TrialFunction(V)

