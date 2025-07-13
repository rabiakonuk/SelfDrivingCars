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

    # TODO: Tune the parameters for QP
    lam = 100  # Tune for tracking
    alpha = 0.1  # Tune for safety constraints
    w = 1000  # Large weight for slack variable
    h = 50  # TODO: Define appropriate value
    B = 20  # TODO: Define appropriate value
    
    # TODO: Complete the matrices and vectors
    P = np.array([[1 / (w**2), 0], [0, 1]])
    
    A = np.array([
        [-2 * m * (x[1] - vd), 0],
        [1 / m * (1.8 + x[1] - vd), -1],
        [0, -Cdg],
        [0, Cag],
        [0, -1]
    ])

    b = np.array([-lam * m * (x[1] - vd), alpha * B, 0, Cag, 0])

    q = np.array([0, w])

    return A, b, P, q

def CarModel(t, x, Student_Controller, param):
    
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]
    
    A, b, P, q = Student_Controller(t, x, param)
    
    var = cp.Variable(2)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                     [A @ var <= b])
    prob.solve()
    
    u = var.value[0]        
    u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])
    
    dx = np.array([param["v0"] - x[1], 
                   u / param["m"]])
    return dx

def sim_vehicle(Student_Controller, param, y0):
    t0, t1 = 0, param["terminal_time"]
    t = np.linspace(t0, t1, 200)
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0
    r = integrate.ode(lambda t, x: CarModel(t, x, Student_Controller, param)).set_integrator("dopri5")
    r.set_initial_value(y0, t0)
    
    # Number of iterations for parameter tuning
    num_iterations = 5
    
    for iteration in range(num_iterations):
        # Run simulation with current parameters
        for i in range(1, t.size):
            y[i, :] = r.integrate(t[i])
            if not r.successful():
                raise RuntimeError("Could not integrate")
        
        # Adjust parameters based on analysis (you can customize this part)
        lam *= 0.9  # Example adjustment - tune this based on your analysis
        param["alpha"] *= 1.1  # Example adjustment
        # Adjust other parameters as needed...
        
        # Reset initial conditions for the next iteration
        r.set_initial_value(y0, t0)

    # Final simulation with tuned parameters
    for i in range(1, t.size):
        y[i, :] = r.integrate(t[i])
        if not r.successful():
            raise RuntimeError("Could not integrate")

    # Recover control input
    u = np.zeros((200, 1))
    for k in range(200):
        if t[k] <= param["switch_time"]:
            param["v0"] = param["v01"]
        if t[k] > param["switch_time"]:
            param["v0"] = param["v02"]
            
        A, b, P, q = Student_Controller(t[k], y[k, :], param)
        var = cp.Variable(2)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(var, P)+ q.T @ var),
                         [A @ var <= b])
        prob.solve()

        u[k] = var.value[0]

    v0 = t * 0
    v0[t < param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg

    return t, B, y, u

param = {"vd": 15.0, "v0": 10.0, "m": 2000, "Cag": 0.3 * 9.81, "Cdg": 0.8 * 9.81, "v01": 5, "v02": 10, "switch_time": 30, "terminal_time": 50}
y0 = np.array([250, 20])
t, B, y, u = sim_vehicle(Student_Controller, param, y0)

# Plot results
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
plt.title('vehicle Distance')
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
plt.title('vehicle velocity')
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
