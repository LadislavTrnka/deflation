from dolfin import *
import deflated_newton
import save_read
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

comm = MPI.comm_world
rank = MPI.rank(comm)
set_log_level(LogLevel.INFO if rank==0 else LogLevel.INFO)
parameters["std_out_all_processes"] = False

# Mesh "right/left"
#mesh = RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "crossed")
#mesh = RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "right/left")
mesh = RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50)
mesh.init()

# Build function spaces
V = VectorFunctionSpace(mesh, "CG", 1)

# Boundaries
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundary_parts.set_all(0)

# Left side
class BoundaryL(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS) 
bL = BoundaryL()
bL.mark(boundary_parts, 1)  
# Right side
class BoundaryR(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, DOLFIN_EPS)
bR = BoundaryR()
bR.mark(boundary_parts, 2) 

u_R = Constant((0.0,0.0))
bndryl = DirichletBC(V, Constant((0.0,0.0)), boundary_parts, 1)
bndryr = DirichletBC(V, u_R, boundary_parts, 2)

bcs = [bndryl,bndryr]

# Initial Guess
u_0 = Function(V)

def Total_energy(u):
    return Energy(u)*dx

def Energy(u):
    # Kinematics
    I = Identity(2)             # Identity tensor
    F = I + grad(u)             # Deformation gradient
    C = F.T*F                   # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Elasticity parameters
    E = 1000000.0
    nu = 0.3
    lmbda = nu*E/(1-nu-2*nu**2)
    mu = E/(2*(1+nu))

    # Stored strain energy density (compressible neo-Hookean model)
    W = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    return W

def residual(u, u_):
    B = Constant((0.0, -1000)) # Body force per unit volume
    Energy = Total_energy(u) - dot(B, u)*dx
    F = derivative(Energy, u, u_)
    return F

def residual_Multiplier(u, u_, p): # Residal with lagrange multiplier p
    B = Constant((0.0, -1000)) # Body force per unit volume
    W = Energy(u)
    E = p*(W - dot(B, u))*dx
    F = derivative(E, u, u_)
    return F

# Test explicit implementation of Newton
def do_test():
    u_R.assign(Constant((-0.06,0.0)))
    u_1 = Deflated_Newton.Newton(residual, u_0, bcs)
    # test
    (u_1_test,flag) = Deflated_Newton.Deflated_Newton(False, residual, u_0, bcs, [])
    # Compute error in L2 norm
    error_L2 = errornorm(u_1, u_1_test, 'L2')
    info(f'error_L2  = {error_L2}')
    return None

# Initial continuation of main branch
def do_continuation(param, u_0, results, results_energy, number_solutions):
    u = Function(u_0.function_space())
    u.assign(u_0)
    for par in param:
        name = "elasticity/"+str(par)+"/"+str(number_solutions[str(par)])
        u_R.assign(Constant((-par,0.0)))
        u_sol = Deflated_Newton.Newton(residual, u, bcs)
        u.assign(u_sol)
        save_read.save_hdf(name, mesh, bd=None, sd=None, sol=u)
        results[str(par)].append(functional(u))
        results_energy[str(par)].append(functional_energy(u))
        number_solutions[str(par)] = number_solutions[str(par)] + 1
    return None

def do_deflation(par, bcs, u_0, results,results_energy, number_solutions):
    roots = []
    for k in range(number_solutions[str(par)]):
        name = "elasticity/"+str(par)+"/"+str(k)
        u_found = save_read.read_hdf(V, name)
        roots.append(u_found)
    i = float(number_solutions[str(par)])
    proceed_with_continuation = 0
    u_R.assign(Constant((-par,0.0)))
    while True:
        info(f"Number of solutions {i}")

        # Sherman–Morrison formula
        #(u_def, flag) = Deflated_Newton.Deflated_Newton(True, residual, u_0, bcs, roots)

        # Lagrange multiplier
        (u_def, flag) = Deflated_Newton.Deflated_Newton_Multiplier(boundary_parts, residual_Multiplier, u_0, u_R, roots)

        if flag==0:
            info(f"Deflation method finds {i} solutions")
            return proceed_with_continuation
        roots.append(u_def)
        results[str(par)].append(functional(u_def))
        results_energy[str(par)].append(functional_energy(u_def))
        name = "elasticity/"+str(par)+"/"+str(number_solutions[str(par)])
        number_solutions[str(par)] = number_solutions[str(par)] + 1
        save_read.save_hdf(name, mesh, bd=None, sd=None, sol=u_def)
        i += 1.0
        proceed_with_continuation = 1 

def do_continuation_forwards(param, k, results,results_energy, number_solutions, branch_no):
    for j in range(branch_no,number_solutions[str(param[k])]):
        name = "elasticity/"+str(param[k])+"/"+str(j)
        u_found = save_read.read_hdf(V, name)
        u = Function(u_found.function_space())
        u.assign(u_found)
        try:
            for par in param[k+1:]:
                name = "elasticity/"+str(par)+"/"+str(number_solutions[str(par)])
                u_R.assign(Constant((-par,0.0)))
                u_sol = Deflated_Newton.Newton(residual, u, bcs)
                u.assign(u_sol)
                results[str(par)].append(functional(u)) 
                results_energy[str(par)].append(functional_energy(u))        
                save_read.save_hdf(name, mesh, bd=None, sd=None, sol=u)
                number_solutions[str(par)] = number_solutions[str(par)] + 1
        except:
            info("Unsuccessful continuation")
        branch_no += 1
    return branch_no

def functional(u):
    # Dispacement at specific point (0.25, 0.05)
    return parallel_eval_vec(u, (0.25, 0.05), 1)

def functional_energy(u):
    return assemble(Total_energy(u))

def parallel_eval_vec(vec, x, component):
    bb = mesh.bounding_box_tree()
    p = Point(x)
    values = np.zeros(mesh.topology().dim())
    ic = 0
    cf = bb.compute_first_entity_collision(p)
    inside = cf < mesh.num_cells()
    if inside :
        vec.eval_cell(values,x,Cell(mesh,cf))
        ic = 1
    comm=MPI.comm_world
    v= MPI.sum(comm, values[component]) / MPI.sum(comm, ic)
    return v

def make_bifurcation_diagram(name, results):
    data_x = []
    data_y = []
    for item in results:
        for i in results[item]:
            data_x.append(float(item))
            data_y.append(float(i))
    max_x = np.max(data_x)
    max_y = np.max(data_y)
    min_y = np.min(data_y)
    min_x = np.min(data_x)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(data_x, data_y, 'ko', markersize=2)
    plt.axis([min_x, max_x, min_y, max_y])
    plt.xlabel('Parameter')
    plt.ylabel('Functional')
    plt.grid(linestyle='dotted')
    major_ticks = np.linspace(min_y, max_y, 11)
    minor_ticks = np.linspace(min_y, max_y, 21)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(tick.MaxNLocator(integer=True))
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.5, linestyle='dotted')
    ax.grid(which='major', alpha=0.5, linestyle='dotted')
    plt.savefig("elasticity/bif_diag_"+name+".pdf")
    #plt.show()
    return None

def clear_folder():
    try:
        info("Clearing")
        location = os.getcwd()
        dir = "elasticity"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    except:
        info("Nothing to clear")
    return None

if __name__ == "__main__":

    if rank == 0:
        clear_folder()
    do_test()

    # Initialization
    results = {}
    results_energy = {}
    number_solutions = {}
    params = list(np.arange(0.0, 0.2, 0.001)) + [0.2]
    #params = list(np.arange(0.0, 0.36, 0.01))
    #params = [0.33,0.35] # -> solution without body force
    for par in params:
        results[str(par)] = []
        results_energy[str(par)] = []
        number_solutions[str(par)] = 0

    # Continuation of main branch
    do_continuation(params, u_0, results, results_energy, number_solutions)

    # Deflated continuation
    u_00 = Function(V)
    branch_no = 1
    for k,par in enumerate(params):
        info(f"Parameter is {par}.")
        if number_solutions[str(par)] == 9:
            break
        u_R.assign(Constant((-par,0.0)))
        if par==params[0]:
            continue
        list = []
        # Deflation
        for i in range(number_solutions[str(params[k-1])]):
            name_0 = "elasticity/"+str(params[k-1])+"/"+str(i)
            u_def_0 = save_read.read_hdf(V, name_0)
            u_00.assign(u_def_0)
            forwards_continuation_boolean = do_deflation(par, bcs, u_00, results,results_energy, number_solutions) # If at least one deflation is successful then run continuation forwards
            list.append(forwards_continuation_boolean)
        # Continuation
        if 1 in list:
            branch_no = do_continuation_forwards(params, k, results, results_energy, number_solutions, branch_no)
    
    # Bifurcation diagrams and save results
    if rank == 0:
        make_bifurcation_diagram("point",results)
        make_bifurcation_diagram("energy",results_energy)
        save_read.save_txt("elasticity/results", results)
        save_read.save_txt("elasticity/results_energy", results_energy)
        save_read.save_txt("elasticity/number_of_sol", number_solutions)
