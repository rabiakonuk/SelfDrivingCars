from sim import *
from utils import *
import casadi as ca

# fik:  ddelta_max has never been used, check if I should use it or not

def nmpc_controller(kappa_table = None):

    ## design your own planning horizon. 
    T = 4 # [s]
    N = 40 # number of control intervals
    h = T / N # length of each control interval

    ###################### Modeling Start ######################
    # system dimensions
    Dim_state = 6
    Dim_ctrl  = 2
    Dim_aux   = 4 

    # define the dynamics
    xm = ca.MX.sym('xm', (Dim_state, 1)) # state 
    um = ca.MX.sym('um', (Dim_ctrl, 1)) # control
    zm = ca.MX.sym('zm', (Dim_aux, 1)) # auxiliary variable

    ## rename the control inputs
    Fx = um[0] # traction force Fxk
    delta = um[1] # steering angle delta_k
    Fxf, Fxr = chi_fr(Fx) # traction force distribution

    ## air drag
    Fd = param["Frr"] + param["Cd"] * xm[0]**2 
    Fd = Fd * ca.tanh(- xm[0] * 100) # deal with the discontinuity at 0
    Fb = 0.0 # aerodynamic drag

    # getslipangle: Ux: longitudinal velocity, Uy: lateral velocity, r: yaw rate, delta: steering angle, param: vehicle parameters
    af, ar  = get_slip_angle(xm[0], xm[1], xm[2], delta, param) # slip angle
    Fzf, Fzr = normal_load(Fx, param) # normal load distribution

    Fyf = tire_model_sim(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"])
    Fyr = tire_model_sim(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"])

    dUx  = (Fxf * ca.cos(delta) - Fyf * ca.sin(delta) + Fxr - Fd) / param["m"] + xm[2] * xm[1]
    dUy  = (Fyf * ca.cos(delta) + Fxf * ca.sin(delta) + Fyr + Fb) / param["m"] - xm[2] * xm[0]
    dr   = (param["L_f"] * (Fyf * ca.cos(delta) + Fxf * ca.sin(delta)) - param["L_r"] * Fyr) / param["Izz"] 
    dx   = ca.cos(xm[5]) * xm[0] - ca.sin(xm[5]) * xm[1]
    dy   = ca.sin(xm[5]) * xm[0] + ca.cos(xm[5]) * xm[1]
    # fik: alternatively -- dy = ca.sin(xm[5]) * xm[0] + ca.sin(xm[5]) * xm[1] : from notebook
    dyaw = xm[2]
      
    xdot = ca.vertcat(dUx, dUy, dr, 
                       dx,  dy, dyaw)

    xkp1 = xdot * h + xm # Euler integration 
    Fun_dynmaics_dt = ca.Function('f_dt', [xm, um, zm], [xkp1]) # discrete time dynamics

    # enforce constraints for auxiliary variable z[0] = Fyf
    alg  = ca.vertcat(Fyf - zm[0], Fyr - zm[1], Fxf - zm[2], Fxr - zm[3]) # algebraic constraints
    Fun_alg = ca.Function('alg', [xm, um, zm], [alg]) 
    
    state_ub = np.array([ 1e2,  1e2,  1e2,  1e8,  1e8,  1e8]) ## Ux | Uy | r | x | y | yaw
    state_lb = np.array([-1e2, -1e2, -1e2, -1e8, -1e8, -1e8]) 

    # fik: F_max uydurma tamamen
    ctrl_ub  = np.array([1e3, param["delta_max"]]) ## max traction force & steering angle]
    ctrl_lb  = np.array([-1e3, -param["delta_max"]])
    
    aux_ub   = np.array([ 1e5,  1e5,  1e5,  1e5]) ## Fyf | Fyr | Fxf | Fxr
    aux_lb   = np.array([-1e5, -1e5, -1e5, -1e5]) ## Fyf | Fyr | Fxf | Fxr
    
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)
    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)
    ub_z = np.matlib.repmat(aux_ub, N, 1)
    lb_z = np.matlib.repmat(aux_lb, N, 1)

    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), 
                             lb_x.reshape((Dim_state * (N+1), 1)),
                             lb_z.reshape((Dim_aux * N, 1))
                             ))

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), 
                             ub_x.reshape((Dim_state * (N+1), 1)),
                             ub_z.reshape((Dim_aux * N, 1))
                             ))
    
    ###################### MPC variables ######################
    x = ca.MX.sym('x', (Dim_state, N + 1)) # state trajectory variables
    u = ca.MX.sym('u', (Dim_ctrl, N)) # control trajectory variables
    z = ca.MX.sym('z', (Dim_aux, N)) # auxiliary variables
    p = ca.MX.sym('p', (Dim_state, 1)) # initial condition

    ###################### MPC constraints start ######################

    ## MPC equality constraints ##
    # G(x) = 0

    # initial con
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))
    
    # Initialize the list to hold the initial condition constraint
    cons_init = []

    # Add the initial condition constraint
    cons_init.append(x[:, 0] - p) # x[:,0] contains the initial state which is u_x, u_y, r, x, y, yaw

    cons_dynamics = []
   
    for k in range(N-1):
        # s_k+1 = f(s_k, u_k, z_k)
        xkp1 = Fun_dynmaics_dt(x[:, k], u[:, k], z[:, k]) 

        # 0 = tire model
        Fy2  = Fun_alg(x[:, k], u[:, k], z[:, k]) # fy2 is the list of lsteral forces on f&r tires

        # add tire model constraints for the current time step k
        for j in range(2):
            cons_dynamics.append(Fy2[j])

        # add dynamics constraints 
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] - xkp1[j]) #

    
    ## MPC inequality constraints ##
    # G(x) <= 0
    # frictioncone(s_k, u_k, z_k) <= 0
    # u_min <= u_k <= u_max
    # otherconstraints (s_k, u_k, z_k) <= 0

    cons_ineq = []

    ## state / inputs limits:
    for k in range(N):
        ## minimal longitudinal speed 
        # fik: it was in the reverse order, I've change it (was >=) 
        cons_ineq.append(2.0 - x[0, k]) # minimal longitudinal speed is 2 m/s

        ## engine power limits (<=)
        cons_ineq.append(u[0, k] - (param['Peng'] / x[0, k])) # F_x <= P_eng / U_x
        
        ## collision avoidance (<=)
        cons_ineq.append(1 - ((x[3, k] - 500)/10)**2 - (x[4, k]/10)**2)

        ## steering angle limits
        delta_max = param["delta_max"]  # steering angle limit

        # ddelta_max may be used? why did they give that value?
        cons_ineq.append(u[1, k] - delta_max) # delta_k - delta_max <= 0: RHS
        cons_ineq.append(- u[1, k] - delta_max)  # -delta_max - delta_k  <= 0: LHS
    
        ## friction cone constraints
        # fik: for içerde mi olmalı, yoksa sola indent mi edilmeli?
        # for k in range(N):
        
        Fx    = u[0, k]  # traction force 
        delta = u[1, k]  # steering angle
        Ux = x[0, k]    # longitudinal velocity

        # fik: check this af, ar get_slip_angle(Ux, Uy, r, delta, param)
        af, ar = get_slip_angle(x[0], x[1], x[2], delta, param)  # slip angle
        Fzf, Fzr = normal_load(Fx, param)  # normal load
        Fxf, Fxr = chi_fr(Fx)  # use chi_fr function for traction force distribution
        
        Fyf = tire_model_ctrl(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"]) if k < N-1 else 0.0  # front tire
        Fyr = tire_model_ctrl(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"]) if k < N-1 else 0.0  # rear tire

        # to match the slacked friction cone constraints from the project description
        cons_ineq.append(Fyf**2 + Fxf**2 - (param["mu_f"] * Fzf)**2 - z[:, k][2]**2)
        cons_ineq.append(Fyr**2 + Fxr**2 - (param["mu_r"] * Fzr)**2 - z[:, k][3]**2)
            

    ###################### MPC cost start ######################
    ## cost function design
    # fik: check the index of the state variables
    J = 0.0

    # Add terms to minimize the squared difference between the final state and a target state
    J += 100 * ((x[0, -1] - 10)**2 + (x[1, -1] - 10)**2)  # Final state

    ## Weights for different objectives
    W_y = 10.0  # Weight for lateral error
    W_phi = 1.0  # Weight for heading error
    W_Ux = 5.0  # Weight for longitudinal speed
    W_x = 2.0  # Weight for distance covered

    ## Stay close to the lane and stabilize the yaw angle
    
    for k in range(N):
        
        e_y = x[4, k] 
        e_phi = x[5, k]  
        Ux = x[0, k] 
        x_pos = x[3, k]  

        # Add terms to minimize the squared difference between the state and a target state at each time step
        J += 10 * ((x[0, k] - 10)**2 + (x[1, k] - 10)**2)
        # Add terms to minimize the square of the control inputs at each time step
        J += 1 * (u[0, k]**2 + u[1, k]**2)

        # Lane tracking and yaw stabilization
        # calculates the contribution of the lane tracking and yaw stabilization to the cost function
        J = J + W_y * e_y**2 + W_phi * e_phi**2 

        # Drive fast/far
        # calculates the contribution of the vehicle's speed and position to the cost function
        J = J - W_Ux * Ux - W_x * x_pos
 
        ## excessive slip angle / friction
        Fx = u[0, k]; delta = u[1, k]
        # af, ar = get_slip_angle(x[:, k], u[:, k], param)
        af, ar = get_slip_angle(x[0], x[1], x[2], delta, param)
        Fzf, Fzr = normal_load(Fx, param)
        Fxf, Fxr = chi_fr(Fx)

        xi = 0.85 # friction cone slack variable
        F_offset = 2000 ## a slacked ReLU function using only sqrt()
        Fyf_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2
        Fyf_max_sq = (ca.sqrt( Fyf_max_sq**2 + F_offset) + Fyf_max_sq) / 2
        Fyf_max = ca.sqrt(Fyf_max_sq)

        ## modified front slide sliping angle
        alpha_mod_f = ca.arctan(3 * Fyf_max * xi / param["C_alpha_f"])

        Fyr_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2

        Fyr_max_sq = (ca.sqrt( Fyr_max_sq**2 + F_offset) + Fyr_max_sq) / 2
        Fyr_max = ca.sqrt(Fyr_max_sq)

        ## modified rear slide sliping angle
        alpha_mod_r = ca.arctan(3 * Fyr_max * xi / param["C_alpha_r"])

        ## limit friction penalty
        # Avoid front tire saturation
        J = J + ca.if_else(ca.fabs(af) >= alpha_mod_f, 1e4 * (ca.fabs(af) - alpha_mod_f)**2, 0.0)
        ## Avoid  rear tire saturation
        J = J + ca.if_else(ca.fabs(ar) >= alpha_mod_r, 1e4 * (ca.fabs(ar) - alpha_mod_r)**2, 0.0)

        ## Penalize slack variable for friction cone limits (both front & rear tires)
        J = J + 1e4 * (ca.fabs(Fyf) - param["mu_f"] * Fzf)**2
        J = J + 1e4 * (ca.fabs(Fyr) - param["mu_r"] * Fzr)**2
        


    lb_dynamics = np.zeros((len(cons_dynamics), 1))
    ub_dynamics = np.zeros((len(cons_dynamics), 1))

    lb_ineq = np.zeros((len(cons_ineq), 1)) - 1e9
    ub_ineq = np.zeros((len(cons_ineq), 1))

    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)), z.reshape((Dim_aux * N, 1)))
    cons_NLP = cons_dynamics + cons_ineq + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_ineq, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_ineq, ub_init_cons))

    n_x = vars_NLP.shape[0]
    n_g = cons_NLP.shape[0]

    prob = {"x": vars_NLP, "p":p, "f": J, "g":cons_NLP}

    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], p.shape[0], lb_var, ub_var, lb_cons, ub_cons
