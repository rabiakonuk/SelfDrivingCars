
import numpy as np
from scipy.optimize import minimize
from scipy import integrate
import matplotlib.pyplot as plt

def Student_Controller(t, x, param):
    vd = param["vd"]
    v0 = param["v0"]

    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    v, delta = x[0], x[1]
    param["q"] = np.array([0, 0, 0])  # or initialize with the appropriate values

    # State variables
    D = x[0]
    v = x[1]

    epsilon = 1e-5
    delta = 0  # Slack variable is set to zero in this implementation

    # Tracking and safety constraints
    h = 0.5 * (v - vd)**2  # Tracking constraint
    B = D - 0.5 * (v0 - v)**2 / Cdg - 1.8 * vd * v / Cdg

    # Constraints for SciPy's minimize function
    constraints = [
        {"type": "ineq", "fun": lambda x: B + delta}
    ]

    P = np.eye(1)  # Identity matrix for the quadratic cost term
    initial_guess = [0]

    # Objective function
    def objective(x):
        return 0.5 * np.dot(x, np.dot(P, x))

    result = minimize(objective, initial_guess, constraints=constraints)

    if not result.success:
        print(f"Optimization failed at time {t}. Message: {result.message}")

    u = result.x[0]
    u = np.clip(u, -param["Cdg"] * param["m"], param["Cag"] * param["m"])
    dx = np.array([param["v0"] - x[1], u / param["m"]])
    q = np.array([0, 0, 0])
    return dx, q

def CarModel(t, x, Student_Controller, param):
    if t <= param["switch_time"]:
        param["v0"] = param["v01"]
    if t > param["switch_time"]:
        param["v0"] = param["v02"]

    dx, q = Student_Controller(t, x, param)
    return dx

def sim_vehicle(Student_Controller, param, y0):
    t0, t1 = 0, param["terminal_time"]
    t = np.linspace(t0, t1, 200)
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0
    r = integrate.ode(lambda t, x: CarModel(t, x, Student_Controller, param)).set_integrator("dopri5")
    r.set_initial_value(y0, t0)

    for i in range(1, t.size):
        y[i, :] = r.integrate(t[i])
        if not r.successful():
            raise RuntimeError("Could not integrate")

    v0 = t * 0
    v0[t < param["switch_time"]] = param["v01"]
    v0[t >= param["switch_time"]] = param["v02"]
    Cdg = param["Cdg"]
    B = y[:, 0] - 1.8 * y[:, 1] - 0.5 * (np.clip(y[:, 1] - v0, 0, np.inf))**2 / Cdg

    return t, B, y

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
    "q": np.array([1.0, 2.0, 3.0]).reshape((3, 1))
}
y0 = np.array([250, 20])
t, B, y = sim_vehicle(Student_Controller, param, y0)

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
