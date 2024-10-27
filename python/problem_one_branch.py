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

def one_branch_search(test_param):
    # Mesh "right/left"
    mesh = test_param["mesh"]
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

    def residual_Multiplier(u, u_, p):
        B = Constant((0.0, -1000)) # Body force per unit volume
        W = Energy(u)
        E = p*(W - dot(B, u))*dx
        F = derivative(E, u, u_)
        return F

    def do_continuation(par, u_0, results, results_energy, test_name):
        u = Function(u_0.function_space())
        u.assign(u_0)
        u_sol = Deflated_Newton.Newton(residual, u, bcs)
        results[str(par)].append(functional(u_sol))
        results_energy[str(par)].append(functional_energy(u_sol))
        return u_sol

    def do_deflation(par, bcs, u_def, u_0, results,results_energy, test_name):
        roots = [u_def]
        i = 1.0
        no_flag = 0
        while True:
            if test_param["deflation"] == "Lagrange":
                (u_def, flag) = Deflated_Newton.Deflated_Newton_Multiplier(boundary_parts, residual_Multiplier, u_0, u_R, roots)
            else:
                (u_def, flag) = Deflated_Newton.Deflated_Newton(True, residual, u_0, bcs, roots)
            if flag==0:
                info(f"Deflation method finds {i} solutions")
                break
            roots.append(u_def)
            no_flag += flag
            results[str(par)].append(functional(u_def)) 
            results_energy[str(par)].append(functional_energy(u_def)) 
            i += 1.0
        return no_flag

    def functional(u):
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

    def make_bifurcation_diagram(folder, name, results):
        data_x = []
        data_y = []
        for item in results:
            for i in results[item]:
                data_x.append(float(item))
                data_y.append(float(i))
        max_x = np.max(data_x)
        max_y = 0.1
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
        plt.savefig(folder+"/bif_diag_"+name+".pdf")
        #plt.show()
        return None

    test_name = test_param["name"]
    def clear_folder():
        try:
            info("Clearing")
            location = os.getcwd()
            dir = "elasticity"+test_name
            path = os.path.join(location, dir)
            shutil.rmtree(path)
        except:
            info("Nothing to clear")
        return None

    if rank == 0:
        clear_folder()

    # Initialization
    results = {}
    results_energy = {}
    params = list(np.arange(0.0, 0.2, 0.001)) + [0.2]
    #params = list(np.arange(0.0, 0.36, 0.01))
    #params = [0.33,0.35] # -> solution without body force
    for par in params:
        results[str(par)] = []
        results_energy[str(par)] = []

    # Deflation from main branch
    i = 0
    no_flag = 0
    u_00 = Function(V)
    for par in params:
        info(f"Parameter is {par}.")
        u_R.assign(Constant((-par,0.0)))
        if par==params[0]:
            # Initial continuation
            u_1 = do_continuation(par, u_0, results, results_energy, test_name)
            u_0.assign(u_1)
            continue
        # Continuation
        u_2 = do_continuation(par, u_0, results, results_energy, test_name)
        u_00.assign(u_2) # found root
        no_flag_single = do_deflation(par, bcs, u_00, u_0, results,results_energy, test_name)
        u_0.assign(u_2) # next initial guess
        no_flag += no_flag_single
        i += 1
    
    # Bifurcation diagrams and save results
    if rank == 0:
        test_name = test_param["name"]
        os.mkdir("elasticity"+ test_name)
        save_read.save_txt("elasticity"+test_name+"/results", results)
        save_read.save_txt("elasticity"+test_name+"/flag", no_flag)
        save_read.save_txt("elasticity"+test_name+"/results_energy", results_energy)
        make_bifurcation_diagram("elasticity"+test_name,"point",results)
        #make_bifurcation_diagram("elasticity"+test_name,"energy",results_energy)

    return None

if __name__ == "__main__":
    testA = {
    "deflation": "Sherman",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "crossed"),
    "name": "A"
    }  

    testB = {
    "deflation": "Sherman",
    "mesh":  RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "right/left"),
    "name": "B"
    }  

    testC = {
    "deflation": "Sherman",
    "mesh":  RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "right"),
    "name": "C"
    }  

    testD = {
    "deflation": "Sherman",
    "mesh":  RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "left"),
    "name": "D"
    }  

    testE = {
    "deflation": "Sherman",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 100, 100, "right"),
    "name": "E"
    }  

    testE1 = {
    "deflation": "Sherman",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 150, 150, "right"),
    "name": "E1"
    }  

    testF = {
    "deflation": "Lagrange",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "crossed"),
    "name": "F"
    }  

    testG = {
    "deflation": "Lagrange",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 50, 50, "right"),
    "name": "G"
    }  

    testH = {
    "deflation": "Sherman",
    "mesh": RectangleMesh(Point(0, 0), Point(1, 0.1), 20, 20),
    "name": "H"
    }  

    test = [testA,testB,testC,testD,testE,testE1,testF,testG,testH]
    for i in test:
        one_branch_search(i)
        