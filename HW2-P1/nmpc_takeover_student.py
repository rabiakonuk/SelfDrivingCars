import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    h = 0.1
    N = int(np.ceil(5/ h)) 

    # System dimensions 
    # 4 states: x, y, yaw, v
    # 2 control inputs: a, delta
    Dim_state = 4  # Number of state variables
    Dim_ctrl = 2  # Number of control variables

    # Additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1))  # initial condition, # the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1))      # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')                  # desired speed of ego car
    delta_last = ca.MX.sym('delta_last')        # steering angle at last step
    # concatenate them
    par = ca.vertcat(x_init, v_leader, v_des, delta_last) 
    
    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1)) # state
    u_model = ca.MX.sym('um', (Dim_ctrl, 1)) # control input

    L_f = 1.0 # Car parameters, do not change, L_f is the distance between the front axle and the center of gravity
    L_r = 1.0 # Car parameters, do not change, L_r is the distance between the rear axle and the center of gravity

    beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1])) # slip angle calculated from: beta = arctan(L_r / (L_r + L_f) * arctan(delta))
    xdot = ca.vertcat(
        x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0], # xdot = v * cos(psi + beta) - v_leader[0]
        x_model[3] * ca.sin(x_model[2] + beta), # ydot = v * sin(psi + beta)
        x_model[3] / L_r * ca.sin(beta), # psidot = v / L_r * sin(beta)
        u_model[0] # vdot = a
    ) 

    # Discrete time dynamics model 
    # fik: added [:4] , might be wrong, check it later
    # Fun_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, par], [xdot * h + x_model[:4]])
    Fun_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, par], [xdot * h + x_model])

    # Declare model variables, note the dimension
    x = ca.MX.sym('x', (Dim_state, N+1))
    u = ca.MX.sym('u', (Dim_ctrl, N))
    # print(x.shape)
    
    # Define weights for each term in the cost function
    w1 = 1e3  # Weight for tracking desired longitudinal velocity
    w2 = 1e5  # Weight for regularizing lateral velocity and yaw rate
    w3 = 1e3  # Weight for lane-keeping
    w4 = 1e1  # Weight for regularizing control inputs

    # Keep in the same lane and take over it while maintaing a high speed
    # Terminal cost
    
    P = w1 * (x_model[3] - v_des)**2  # Objective C1: tracking desired speed
    print("v_des: ", v_des)
    # Running cost
    L = w4 * ca.mtimes(u_model.T, u_model)  # Objective C4: regularization on control inputs  

    L += w2 * (x_model[1]**2 + x_model[2]**2) # Objective C2: regularization on lateral velocity and yaw rate
    
    #fik: y_leader = ca.MX.sym('y_leader') # leader car's y position w.r.t ego car
    #par = ca.vertcat(x_init, v_leader, v_des, delta_last, y_leader)  # Add y_leader to the parameter list
    #L += w3 * (x_model[1] - y_leader)**2  # Objective C3: lane-keeping
    
    lane_center = 1.0  # in meters
    L += w3 * (x_model[1] - lane_center)**2  # Objective C3: lane-keeping

    Fun_cost_terminal = ca.Function('P', [x_model, par], [P])
    Fun_cost_running = ca.Function('Q', [x_model, u_model, par], [L])

    # fik: find ub & lb for state: [x, y, yaw, v]
    # state: [x, y, yaw, v], x in here is the distance between ego car and leader car in x direction
    state_ub = np.array([1000, 3, np.pi, 50])  # Replace with actual upper bounds
    # fik: hıza lb olarak ne koyarsam ona converge ediyor ? 
    state_lb = np.array([1.0, -1, -np.pi, 40])  # Replace with actual lower bounds
    ctrl_ub = np.array([4.0, 0.6])  # Max acceleration and steering angle
    ctrl_lb = np.array([-10.0, -0.6])  # Min acceleration and steering angle

    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)
    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    # check here for the dimension
    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N + 1), 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N + 1), 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    cons_dynamics = []
    ub_dynamics = np.zeros((Dim_state * N, 1))
    lb_dynamics = np.zeros((Dim_state * N, 1))
    for k in range(N):
        Fx = Fun_dynmaics_dt(x[:, k], u[:, k], par)
        cons_dynamics.append(Fx - x[:, k+1])

    # state constraints: G(x) <= 0 
    # update them according to the problem definitions

    cons_state = []
    for k in range(N):
        #### collision avoidance: 
        # cons_state.append(Your collision avoidance constraint here)
        r_x = 30
        r_y = 2
        # Ellipsoidal constraint for collision avoidance
        ellipsoidal_constraint = (x[0, k] / r_x)**2 + (x[1, k] / r_y)**2 - 1
        # Append the constraint
        cons_state.append(ellipsoidal_constraint)

        #### Maximum lateral acceleration ####
        # dx = (x[:, k+1] - x[:, k]) / h 
        dot_psi = (x[2, k+1] - x[2, k]) / h # yaw rate
        ay = x[3, k] * dot_psi # lateral acceleration
        gmu = (0.5 * 0.6 * 9.81)
        cons_state.append(ay - gmu)
        cons_state.append(-ay - gmu)
        # state: x = [x, y, yaw, v]

        #### lane keeping ####
        y_L = 3.0  # in meters
        y_R = -1.0  # in meters
        cons_state.append(x[1, k] - y_L)
        cons_state.append(-x[1, k] + y_R)

        #### steering rate ####
        # max allowable rate of change of steering angle
        delta_max = 0.6
        if k >= 1:
            d_delta = (u[1, k] - u[1, k-1]) / h 
            cons_state.append(d_delta - delta_max)
            cons_state.append(-d_delta - delta_max)
        else:
            d_delta = (u[1, 0] - delta_last) / h
            cons_state.append(d_delta - delta_max)
            cons_state.append(-d_delta - delta_max)
    
    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # cost function: # NOTE: You can also hard code everything here
    J = Fun_cost_terminal(x[:, -1], par)
    for k in range(N):
        J = J + Fun_cost_running(x[:, k], u[:, k], par)

    # initial condition as parameters
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))
    
    # Define variables for NLP solver
    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)))
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_state_cons, ub_init_cons))

    # Create an NLP solver
    prob = {"x": vars_NLP, "p":par, "f": J, "g":cons_NLP}
    
    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], par.shape[0], lb_var, ub_var, lb_cons, ub_cons  