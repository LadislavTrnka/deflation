from dolfin import *
import numpy as np
from ufl import product

def Newton(residual, u_0, bcs):
    # Standard Newton solver
    V = u_0.function_space()
    u = Function(V)
    u_ = TestFunction(V)
    u.assign(u_0)
    F = residual(u,u_)
    J = derivative(F,u)
    
    problem = NonlinearVariationalProblem(F,u,bcs,J)
    solver = NonlinearVariationalSolver(problem)

    # solver.parameters['nonlinear_solver'] = 'snes'
    # solver.parameters['snes_solver']['line_search'] = 'basic'
    # solver.parameters['snes_solver']['linear_solver']= 'lu'

    solver.parameters['newton_solver']['error_on_nonconvergence'] = False
    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 5e-9
    solver.parameters['newton_solver']['relative_tolerance'] = 5e-9
    solver.parameters['newton_solver']['maximum_iterations'] = 20
    solver.solve()
    return u

def Deflated_Newton(deflation, residual, u_0, bcs, roots):
    V = u_0.function_space()
    # Explicit implementation of Newton iterations
    u_k = Function(V)
    w = TestFunction(V)
    u_k.assign(u_0)

    # Tolerance and max iterations
    res_norm = 1.0
    rel_res_norm = 1.0
    if deflation:
        tol = 1.0e-7
        rel_tol = 0.0
    else:
        tol = 5e-9
        rel_tol = 5e-9      
    iter = 0
    maxiter = 100

    # Residual
    F = residual(u_k,w)
    a = derivative(F,u_k)
    L = -F
    
    du_Newton = Function(V)
    flag = 0

    bcs_Newton = []

    ## Homogenize boundary conditions
    # for bc in bcs:    # doesn't work so well -> nan
    #     bc.apply(u_k.vector())
    #     bc.homogenize()
    # bcs_Newton = bcs

    bcs_vec = []
    bndry_Functions = []
    for bc in bcs:
        bcs_vec.append(interpolate(bc.value(),V))   # Extract boundary conditions
        bcs_Newton.append(DirichletBC(bc))          # Copy bcs
        bndry_Functions.append(Function(V))         # For each bc in bcs create update fct

    while iter < maxiter:

        # Dirichlet boundary conditions du_Newton|_bndry = u_Dirichlet - u_k
        for (bc,bc_vec, bndry) in zip(bcs_Newton, bcs_vec, bndry_Functions):
            bndry.vector()[:] = bc_vec.vector() - u_k.vector()
            bc.set_value(bndry)

        # Assemble matrix and RHS
        ## Option 1 
        (A, b) = assemble_system(a, L, bcs_Newton)
        try:
            solve(A, du_Newton.vector(), b)
        except:
            info("Unable to solve linear system")
            break

        ## Option 2
        # problem=LinearVariationalProblem(a,L,du_Newton,bcs)
        # solver=LinearVariationalSolver(problem)
        # solver.parameters['linear_solver']='mumps'
        # try:
        #     solver.solve()
        # except:
        #     info("Unable to solve linear system")
        #     break        
        # b = assemble(F)
        # for bc in bcs:
        #     bc.apply(b)

        # Check Nan
        if str(du_Newton.vector().norm("l2")) == "nan":
            info("NaN Error")
            break

        # Compute norms
        res_norm = b.norm('l2')
        if iter == 0:
            resnorm_0 = res_norm
        rel_res_norm = res_norm/resnorm_0

        info(f'Iter: {iter}, Norm: {res_norm}, Rel_Norm: {rel_res_norm}')       
        
        # Stop criteria
        if res_norm < tol or rel_res_norm < rel_tol:
            flag = 1
            info("Naive Newton converged.")
            return (u_k, flag)
        
        # Step
        if deflation:       # Deflated step
            (eval, der) = Deflation_operator(V, u_k, roots)
            factor = float(1-eval**(-1)*der.vector().inner(du_Newton.vector()))
            factor2 = float(1/factor)
            u_k.vector()[:] = u_k.vector() + factor2*du_Newton.vector()
        else:    # Newton step
            u_k.vector()[:] = u_k.vector() + du_Newton.vector()
        iter += 1
    info("Naive Newton did not converged.")
    return (u_k, flag)

def Deflated_Newton_Multiplier(boundary_parts, residual_Multiplier, u_0, u_R, roots):
    V = u_0.function_space()
    mesh = V.mesh()
    n = len(roots)

    # Adding real elemets
    Ev = VectorElement("CG", mesh.ufl_cell(),1)
    R = FiniteElement("Real", mesh.ufl_cell(),0)
    Z = FunctionSpace(mesh, MixedElement([Ev]+n*[R]))

    w = Function(Z)
    w_ = TestFunction(Z)

    # Initial guesses
    assign(w.sub(0), u_0)
    for i in range(n):
        assign(w.sub(i+1), interpolate(Constant(100000),Z.sub(1).collapse()))

    # Parameters of the deflation operator
    power = 2.0
    shift = 1.0
    eval = 1.0
    
    # Construction of residual
    F = 0
    for i,root in enumerate(roots):
        wTUPLE = split(w)
        w_TUPLE = split(w_)
        u = wTUPLE[0]
        u_ = w_TUPLE[0]
        F = F + residual_Multiplier(u,u_,wTUPLE[i+1]) #residual_Multiplier(u,u_,p)
        F = F + squared_norm_p(u, root, w_TUPLE[i+1]*(wTUPLE[i+1]-shift)) - w_TUPLE[i+1]*dx   #squared_norm_p(u, root, p_*(p-shift)) - p_*dx

    J = derivative(F,w)

    bndryl = DirichletBC(Z.sub(0), Constant((0.0,0.0)), boundary_parts, 1)
    bndryr = DirichletBC(Z.sub(0), u_R, boundary_parts, 2)
    bcs = [bndryl,bndryr]
    
    problem = NonlinearVariationalProblem(F,w,bcs,J)
    solver = NonlinearVariationalSolver(problem)

    solver.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver.parameters['newton_solver']['absolute_tolerance'] = 5e-9
    solver.parameters['newton_solver']['relative_tolerance'] = 5e-9
    solver.parameters['newton_solver']['maximum_iterations'] = 100

    # Solver solve
    flag = 0
    try: 
        solver.solve()
        flag = 1
    except:
        info("An exception occurred")

    u = w.sub(0, deepcopy=True)
    return (u, flag)

def squared_norm(a, b):
    return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

def squared_norm_p(a, b, p):
    return p*inner(a - b, a - b)*dx + p*inner(grad(a - b), grad(a - b))*dx

def Deflation_operator(V, u, roots):
    power = 2.0
    shift = 1.0
    m = 1.0
    for root in roots:
        normsq = assemble(squared_norm(u, root))
        scalar = normsq**(-power/2.0) + shift
        m *= scalar
    der = Function(V)
    factors  = []
    dfactors = []
    dnorms = []
    norms  = []
    for root in roots:
        form = squared_norm(u, root)
        norms.append(assemble(form))
        dnorms.append(assemble(derivative(form, u)))
    for normsq in norms:
        factor = normsq**(-power/2.0) + shift
        dfactor = (-power/2.0) * normsq**((-power/2.0) - 1.0)
        factors.append(factor)
        dfactors.append(dfactor)
    eta = product(factors)
    for (solution, factor, dfactor, dnormsq) in zip(roots, factors, dfactors, dnorms):
        der.vector()[:] = der.vector() + (float(eta/factor)*dfactor)*dnormsq
    return (m, der)

    