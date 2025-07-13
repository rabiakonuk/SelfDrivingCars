import cvxpy as cp
import numpy as np
from scipy import integrate

# Debugging flag to disable some constraints
debug_disable_constraints = False
# Define the QP solver function

def qp_solver(P, q, A, b):
    x = cp.Variable(q.shape[0])
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
    constraints = [A @ x <= b]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Problem status: {prob.status}")
        return None
    
    return x.value

def Student_Controller(t, x, param):
    import numpy as np

    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]
    g = 9.81  # gravity in m/s^2

    # Define the parameters for the QP problem
    lam = 1e2  # For tracking, larger values for faster tracking 20 mantıklı olan ama 25 kalpten
    alpha = 0.22 # For safety, smaller values for slower convergence to B=0
    w = 1e7  # Large weight for the slack variable
    delta = 1

    # Calculate h and B based on the project description
    h = 0.5 * (x[1] - vd)**2 # h = 0.5 * (v - vd)^2
    B = x[0] - 0.5 * ((v0 - x[1])**2) / (Cdg) - 1.8 * x[1] # B = x[0] - 0.5 * ((v0 - v)^2) / (Cdg * g) - 1.8 * v

    # Define the quadratic term of the objective function
    P = np.array([[1.0, 0.0], [0.0, w]]) # P = [[1, 0], [0, w]]

    # Define the linear term of the objective function
    q = np.zeros([2, 1]) # q = [0, 0]
    # fik bunsuz dene bir de 
    # q[0] = Cdg*m # Set the first element of q to Cd

    # Define the constraint matrix A and vector b
    A = np.zeros([5, 2]) 
    b = np.zeros([5]) 

    # Constraint 1: (v - vd)/m * Fw <= -lam * h + delta
    A[0, 0] = (x[1] - vd) / m # A[0, 0] = (v - vd)/m
    A[0, 1] = -delta  # Include delta in the constraint 
    b[0] = -lam * h 

    # fik: (v0 - v) kısmı taşınabilir
    # Constraint 2: (v0 - v) - (1.8 + (v - v0)/(Cdg * g))/m * Fw >= -alpha * B
    A[1, 0] = ((1.8 + (x[1] - v0) / (Cdg)) / m) 
    b[1] = alpha * B + (v0 - x[1])

    # Constraint 3: -Cdg * g <= Fw/m
    A[2, 0] = -1 / m
    b[2] = Cdg

    # Constraint 4: Fw/m <= Cag * g
    A[3, 0] = 1 / m
    b[3] = Cag

    # Constraint 5: delta >= 0 
    A[4, 1] = -delta 
    b[4] = 0.0

        # Call the QP solver to get the optimal control input and slack variable
    var = qp_solver(P, q, A, b)
    
    if var is None:  # Check if the QP solver returned a valid solution
        print(f"Debug Info at time {t}:")
        print(f"h = {h}, B = {B}")
        print(f"A = {A}, b = {b}")
        print(f"P = {P}, q = {q}")
        F_w = 0  # Set a default value or you can raise an exception
    else:
        F_w = var[0]  # Assuming F_w is the first element in the solution vector
    
    return A, b, P, q


def CarModel(t, x, Student_Controller, param, qp_solver):
    
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]
    
    A, b, P, q = Student_Controller(t, x, param)

    var = qp_solver(P, q, A, b)  # Assuming qp_solver is your QP solving function
    if var is not None:
        u = var[0]
    else:
        print(f"Debug Info at time {t}:")
        print(f"A = {A}, b = {b}")
        print(f"P = {P}, q = {q}")
        u = 0  # Or some safe default value
    
    var = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                     [A @ var <= b])
    prob.solve()
    
    u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])
    
    dx = np.array([param["v0"] - x[1], 
                   u / param["m"]])
    return dx

def sim_vehicle(Student_Controller, param, y0):
    
    t0, t1 = 0, param["terminal_time"]                # start and end
    t = np.linspace(t0, t1, 200)  # the points of evaluation of solution
    # y0 = [250, 10]                   # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0

    r = integrate.ode(lambda t, x: CarModel(t, x, Student_Controller, param, qp_solver)).set_integrator("dopri5")
    r.set_initial_value(y0, t0)   # initial values
    for i in range(1, t.size): 
       y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
       if not r.successful():
           raise RuntimeError("Could not integrate")
    
    ### recover control input ###
    u = np.zeros((200, 1))
    for k in range(200):
        if t[k] <= param["switch_time"]:
            param["v0"] = param["v01"]
        if t[k] > param["switch_time"]:
            param["v0"] = param["v02"]
        if t[k] == 0:
            u[k] = (0 - param["Cdg"]) * param["m"]
        
        A, b, P, q = Student_Controller(t[k], y[k, :], param)
        var = qp_solver(P, q, A, b)
        
        if var is not None:
            u[k] = var[0]
        else:
            print(f"QP problem is infeasible at time {t[k]}")
            u[k] = 0  # Or some safe default value

    ### recover control input ###

    v0 = t * 0
    v0[t <  param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B   = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg
    if t[0] == 0:
        u[0] = (0 - param["Cdg"]) * param["m"]

    return t, B, y, u

