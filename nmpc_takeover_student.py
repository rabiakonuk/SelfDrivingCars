import casadi as ca
import numpy as np
import casadi as ca
import time
from numpy import linalg as LA

def nmpc_controller():
    # Declare simulation constants
    T = 2
    N = 20 # TODO  You are supposed to design the planning horizon
    h = 0.1 # TODO: What is the time interval for simulation? 

    # system dimensions
    Dim_state = 4 # TODO
    Dim_ctrl  = 2 # TODO

    # additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1)) # initial condition, # the state should
                                                    #be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1))      # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')
    delta_last = ca.MX.sym('delta_last')
    par = ca.vertcat(x_init, v_leader, v_des, delta_last)

    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0 # Car parameters, do not change
    L_r = 1.0 # Car parameters, do not change

    beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1])) # TODO 

    xdot = ca.vertcat(x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0],
                      x_model[3] * ca.sin(x_model[2] + beta) - v_leader[1],
                      x_model[3] / L_r * ca.sin(beta),
                      u_model[0]) # TODO

    # Discrete time dynmamics model
    Fun_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, par], [xdot * h + x_model])# TODO
    
    # Declare model variables, note the dimension
    x = ca.MX.sym('x', (Dim_state, N + 1)) # TODO
    u = ca.MX.sym('u', (Dim_ctrl , N)) # TODO

    # Keep in the same lane and take over it while maintaing a high speed
    ### P = 100*(x_model[3]-v_des)**2 +            # TODO
    #### L = # TODO

    #Fun_cost_terminal = ca.Function('P', [x_model, par], [P])
    #Fun_cost_running = ca.Function('Q', [x_model, u_model, par], [L])

    # state and control constraints
    state_ub = np.array([ 1e4,  3,  1e4,  1e4]) #TODO 
    state_lb = np.array([ -1e4, -1, -1e4, -1e4])# TODO 
    ctrl_ub  = np.array([ 4,  0.6])# TODO 
    ctrl_lb  = np.array([-10,  -0.6])# TODO 
    
    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N+1), 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N+1), 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    cons_dynamics = []
    ub_dynamics = np.zeros((N * Dim_state, 1))
    lb_dynamics = np.zeros((N * Dim_state, 1))
    
    for k in range(N):
        Fx = Fun_dynmaics_dt(x[:, k], u[:, k], par)
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] -  Fx[j]) # TODO


    # state constraints: G(x) <= 0
    
    cons_state = []
    for k in range(N):
        #### collision avoidance:
        # TODO
        cons_state.append(- ((x[0, k]/30)**2) - ((x[1, k]/2)**2) + 1)

        #### Maximum lateral acceleration ####
        dx = (x[:, k+1] - x[:, k]) / h
        ay = x[3,k] * dx[2] 
        #ay=v_des*dx[2]
        # TODO: Compute the lateral acc using the hints
        
        gmu = (0.5 * 0.6 * 9.81)
        temp=ca.fabs(ay)
        # condition=ay<0
        #if condition:
        #   ay=-1*ay
        #yy=np.float32(0.0)
        #yy=ay
        #yy=np.abs(yy)
        
        cons_state.append(temp-gmu)
        #cons_state.append(# TODO)

        #### lane keeping ####
        cons_state.append( x[1,k]-3)
        cons_state.append(-(x[1,k])-1)

        #### steering rate ####
        if k >= 1:
            d_delta = (u[1, k] - u[1, k-1]) / h # TODO

            cons_state.append((ca.norm_2(d_delta))- 0.6 )
            cons_state.append( ca.fabs(d_delta)-0.6)
        else:
            d_delta = delta_last # TODO, for the first input, given d_last from param
            cons_state.append((ca.norm_2(d_delta))-0.6)
            cons_state.append(ca.fabs(d_delta)-0.6)

    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # Adding yaw rate regularization for C2 and lane deviation penalty for C3
    yaw_rate_penalty = x[2, :]**2  #  x[2, :] represents yaw rate
    lane_deviation_penalty = (x[1, :] - 3)**2  #  x[1, :] represents lateral position

    # cost function: # NOTE: You can also hard code everything here
    #J = Fun_cost_terminal(x[:, -1], par)
    J=  1e2 * ((x[3, -1] - v_des)**2) + 1e1 * (x[1, -1])**2 + 0.1 * yaw_rate_penalty[-1] +  1e1 * lane_deviation_penalty[-1] # penalize the speed error and lateral error
    
    for k in range(N):
        J += 1e1 * ((x[3, k] - v_des)**2) + 1e3 * ((x[1, k])**2) +0.1 * yaw_rate_penalty[k] + 1e1 * lane_deviation_penalty[k]
        J += 10 * (u[0, k]**2) + 10 * (u[1, k])**2
    
    J = ca.substitute(J, v_des, 50)


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