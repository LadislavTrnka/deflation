from dolfin import *
import problem
import save_read

def save_xdmf(output_dir,param):
    # Create files for storing solution
    ufile = XDMFFile("%s/displacement.xdmf" % output_dir)
    for i in range(9):
        name = "elasticity/"+str(param)+"/"+str(i)
        u = save_read.read_hdf(Problem.V, name)
        u.rename("u", "displacement")
        ufile.write(u, i)

if __name__ == "__main__":
    param = 0.198 # 9 solutions
    output_dir='elasticity/viz'
    save_xdmf(output_dir,param)

