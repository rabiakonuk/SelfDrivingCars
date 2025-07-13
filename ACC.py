import cvxpy as cp
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def Student_Controller(t, x, param):
    vd = param["vd"]
    v0 = param["v0"]

    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    v, delta = x[0], x[1]
    param["q"] = np.array([0, 0])  # or initialize with the appropriate values

    # Assuming param["q"] is a 1D array, you might want to reshape it to (3, 1)
    param["q"] = param["q"].reshape((2, 1))

    # Tuning parameters
    lam = 1.5  # Tune this for tracking
    alpha = 10  # Tune this for safety constraints
    w = 1e6  # Large weight for the slack variable

    # State variables
    D = x[0]
    v = x[1]

    epsilon = 1e-5
    delta = cp.Variable(1, nonneg=True)  # Introduce slack variable

    # Tracking and safety constraints
    h = 0.5 * (v - vd)**2  # Tracking constraint
    B = D - 0.5 * (v0 - v)**2 / Cdg - 1.8 * vd * v / Cdg
    safety_constraint = B + delta >= 0

    P = np.eye(2)  # Identity matrix for the quadratic cost term
    
    A = np.array([[0.5 * (v - vd) / m, 0],  # Constraint for tracking
                  [v - v0 - (1 / m) * (1.8 + v - v0 * Cdg), 0],  # Constraint for safety
                  [0, 0],  # Placeholder for other constraints
                  [0, 0],
                  [0, 0]])

    epsilon = 1e-5
    b = np.array([-lam * h, -alpha * B + epsilon, 0, 0, 0])  # Right-hand side of the constraints

    constraints = [A @ cp.vstack([v, delta]) <= cp.reshape(b, (A.shape[0], 1)), safety_constraint]

    q = np.array([0, 0]).reshape(-1, 1)  # Make q a column vector with shape (2, 1)
    T = np.array([1, 0]).reshape(1, -1)  # Make T a row vector with shape (1, 2)

    # print the dimensions of the matrices and vectors involved in the "constraints" variable line
    # print("A.shape:", A.shape)
    # print("v.shape:", v.shape)
    # print("delta.shape:", delta.shape)
    # print("b.shape:", b.shape)
    # print("P.shape:", P.shape)
    # print("q.shape:", q.shape)
    # print("T.shape:", T.shape)

    # objective = cp.Minimize((1/2) * cp.quad_form(cp.vstack([v, delta]), P) + w * q @ (cp.vstack([v, delta])).T)
    objective = cp.Minimize((1/2) * cp.quad_form(cp.vstack([v, delta]), P) + w * param["q"].T @ cp.vstack([v, delta]))
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()

        # Debugging Step 1: Check for Infeasibility
        if problem.status == cp.INFEASIBLE:
            print("Infeasible problem at time:", t)
            
            # Debugging Step 2: Debug the Constraints
            print("State variables:", x)
            print("Constraints A:", A)
            print("Constraints b:", b)
        
        if problem.status != cp.OPTIMAL:
            raise RuntimeError("Optimization did not converge")

        u = v.value[0]
        u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])

    except Exception as e:
        print(f"Error in optimization at time {t}: {e}")
        u = 0  # Set a default value when optimization fails
    
    dx = np.array([param["v0"] - x[1], u / param["m"]])
    q = np.array([0, 0, 0])
    return A, b, P, q


def CarModel(t, x, Student_Controller, param):
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]
    
    A, b, P, q = Student_Controller(t, x, param)
    
    q = np.array([0, 0]).reshape(-1, 1)  # Make q a column vector with shape (2, 1)
    T = np.array([1, 0]).reshape(1, -1)  # Make T a row vector with shape (1, 2)

    var = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P) + q.T @ var),
                     [A @ var <= b])

    try:
        prob.solve()

        # Debugging Step 1: Check for Infeasibility
        if prob.status == cp.INFEASIBLE:
            print("Infeasible problem at time:", t)
            
            # Debugging Step 2: Debug the Constraints
            print("State variables:", x)
            print("Constraints A:", A)
            print("Constraints b:", b)
        
        if prob.status != cp.OPTIMAL:
            raise RuntimeError("Optimization did not converge")

        u = var.value[0]
        u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])

    except Exception as e:
        print(f"Error in optimization at time {t}: {e}")
        u = 0  # Set a default value when optimization fails

    dx = np.array([param["v0"] - x[1], u / param["m"]])
    return dx

def sim_vehicle(Student_Controller, param, y0):
    t0, t1 = 0, param["terminal_time"]  # start and end
    t = np.linspace(t0, t1, 200)  # the points of evaluation of the solution
    y = np.zeros((len(t), len(y0)))   # array for the solution
    y[0, :] = y0
    r = integrate.ode(lambda t, x: CarModel(t, x, Student_Controller, param)).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t0)   # initial values

    q = np.array([0, 0]).reshape(-1, 1)  # Make q a column vector with shape (2, 1)
    T = np.array([1, 0]).reshape(1, -1)  # Make T a row vector with shape (1, 2)

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
            
        A, b, P, q = Student_Controller(t[k], y[k, :], param)

        q = np.array([0, 0]).reshape(-1, 1)  # Make q a column vector with shape (2, 1)
        T = np.array([1, 0]).reshape(1, -1)  # Make T a row vector with shape (1, 2) 

        var = cp.Variable(2)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P) + q.T @ var),
                         [A @ var <= b])

        try:
            prob.solve()
            if prob.status != cp.OPTIMAL:
                raise RuntimeError("Optimization did not converge")

            u[k] = var.value[0]
        except Exception as e:
            print(f"Error in optimization at time {t[k]}: {e}")
            u[k] = 0  # Set a default value when optimization fails
    ### recover control input ###

    v0 = t * 0
    v0[t < param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg

    return t, B, y, u

# Main code
param = {
    "vd": 15.0,
    "v0": 10.0,
    "m": 2000,
    "Cag": 0.3 * 9.81,
    "Cdg": 0.8 * 9.81,
    "v01": 5,
    "v02": 10,
    "switch_time": 30,
    "terminal_time": 50,
    "q": np.array([1.0, 2.0]).reshape((2, 1))
}
y0 = np.array([250, 20]) # [D(0), v(0)]
t, B, y, u = sim_vehicle(Student_Controller, param, y0)

# Plotting
plt.figure(figsize=(20, 3))
plt.plot(t, B, label="Value of Barrier Function")
plt.xlabel('$Time (s)$')
plt.ylabel('$B$')
plt.title('Safety Constraints')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(20, 3))
plt.plot(t, y[:, 0], label="Distance to leading vehicle")
plt.xlabel('$Time (s)$')
plt.ylabel('$m$')
plt.title('Vehicle Distance')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(20, 3))
plt.plot(t, y[:, 1], label="Ego vehicle Velocity")
v_lead = t * 0
v_lead[t <= param["switch_time"]] = param["v01"]
v_lead[t >= param["switch_time"]] = param["v02"]
plt.plot(t, v_lead, label="Leading vehicle Velocity")
plt.plot(t, t * 0 + param["vd"], "--", label="Desired Velocity")

plt.xlabel('$Time (s)$')
plt.ylabel('$m.s^{-1}$')
plt.title('Vehicle Velocity')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(20, 3))
plt.plot(t, 0 * t - param["Cdg"], "k--", label="Cag")
plt.plot(t, 0 * t + param["Cag"], "k--", label="Cdg")
plt.plot(t, u / param["m"], label="Normalized Control Input")
plt.xlabel('$Time (s)$')
plt.ylabel('$m.s^{-2}$')
plt.title('Normalized Control Inputs')
plt.legend()
plt.grid()
plt.show()