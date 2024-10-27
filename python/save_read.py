import sys, os
import numpy as np
from dolfin import *

def save_hdf(name, mesh, bd=None, sd=None, sol=None):
    hdf = HDF5File(mesh.mpi_comm(), name+".h5", "w")
    if(mesh) : hdf.write(mesh, "/mesh")
    if(sd) : hdf.write(sd, "/subdomains")
    if(bd) : hdf.write(bd, "/boundaries")
    if(sol) : hdf.write(sol, "/solution")

def read_hdf(V,name):
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), name+'.h5', "r")
    hdf.read(mesh, "/mesh", False)
    if hdf.has_dataset("/subdomains"):
        sd = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
        hdf.read(sd, "/subdomains")
    if hdf.has_dataset("/boundaries"):
        bd = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
        hdf.read(bd, "/boundaries")
    if hdf.has_dataset("/solution"):
        u = Function(V)
        hdf.read(u, "/solution")
    #mesh.init()
    return u

def save_txt(name, *args):
    with open(name+".txt", 'w') as f:
        for arg in args:
            f.write(str(arg) +"\n")
    return None
    