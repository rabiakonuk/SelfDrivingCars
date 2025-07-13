# import casadi as ca
import numpy as np
import cvxpy as cp

def Setup_Derivative(param):
    ## this function is optional

    # Dim_state = 4
    # Dim_ctrl  = 2

    # x_model = ca.MX.sym('xm', (Dim_state, 1))
    # u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    # L_f = param["L_f"]
    # L_r = param["L_r"]

    # beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1]))

    # xdot = ca.vertcat(x_model[3] * ca.cos(x_model[2] + beta),
    #                   x_model[3] * ca.sin(x_model[2] + beta),
    #                   x_model[3] / L_r * ca.sin(beta),
    #                   u_model[0])

    # Jac_dynamics_dt = ca.jacobian(xdot * param["h"] + x_model, ca.veccat(x_model, u_model))
    # Fun_Jac_dt = ca.Function('J_dt', [x_model, u_model], [Jac_dynamics_dt])
    
    def Fun_Jac_dt(x, u, param):

        L_f = param["L_f"]
        L_r = param["L_r"]
        h   = param["h"]

        psi = x[2]
        v   = x[3]
        delta = u[1]
        a   = u[0]
        
        A = np.zeros((4, 4))
        B = np.zeros((4, 2))

        A[0, 0] = 1.0
        A[0, 2] = -h*v*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))
        A[0, 3] = h*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))

        A[1, 1] = 1.0
        A[1, 2] = h*v*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))
        A[1, 3] = h*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r)))

        A[2, 2] = 1.0
        A[2, 3] = (h*np.arctan(delta))/(((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)**(1/2)*(L_f + L_r))
        
        A[3, 3] = 1.0

        B[0, 1] =  -(L_r*h*v*np.sin(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r))))/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)*(L_f + L_r))
        B[1, 1] =   (L_r*h*v*np.cos(psi + np.arctan((L_r*np.arctan(delta))/(L_f + L_r))))/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)*(L_f + L_r))
        B[2, 1] =  (h*v)/((delta**2 + 1)*((L_r**2*np.arctan(delta)**2)/(L_f + L_r)**2 + 1)**(3/2)*(L_f + L_r))
        B[3, 0] = h

        return [A, B]

    return Fun_Jac_dt

def Student_Controller_LQR(x_bar, u_bar, x0, Fun_Jac_dt, param):
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = u_bar.shape[0] * u_bar.shape[1]
    n_x = x_bar.shape[0] * x_bar.shape[1]
    n_var = n_u + n_x

    n_eq  = x_bar.shape[1] * u_bar.shape[0] # dynamics
    n_ieq = u_bar.shape[1] * u_bar.shape[0] # input constraints
    
    # define the parameters
    Q = np.eye(4)  * 1
    R = np.eye(2)  * 10
    Pt = np.eye(4) * 1000
    
    # define the cost function
    np.random.seed(1)
    P = np.zeros((n_var, n_var))
    for k in range(u_bar.shape[0]):
        P[k * x_bar.shape[1]:(k+1) * x_bar.shape[1], k * x_bar.shape[1]:(k+1) * x_bar.shape[1]] = Q
        P[n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1], n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1]] = R
    
    P[n_x - x_bar.shape[1]:n_x, n_x - x_bar.shape[1]:n_x] = Pt
    P = (P.T + P) / 2
    q = np.zeros((n_var, 1))
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    # G = np.zeros((n_ieq, n_var))
    # ub = np.zeros(n_ieq)
    # lb = np.zeros(n_ieq)
    
    # u_ub = np.array([  4,  0.8])
    # u_lb = np.array([-10, -0.8])
    
    for k in range(u_bar.shape[0]):
        AB = Fun_Jac_dt(x_bar[k, :], u_bar[k, :], param)
        A[k * dim_state:(k+1) * dim_state,      k * dim_state:(k+1) * dim_state]       = AB[0] # AB[0:dim_state, 0:dim_state]
        A[k * dim_state:(k+1) * dim_state,  (k+1) * dim_state:(k+2) * dim_state]       = -np.eye(dim_state)
        A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]  = AB[1] # AB[0:dim_state, dim_state:]
        
        # G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]    = np.eye(dim_ctrl)
        # ub[k * dim_ctrl:(k+1) * dim_ctrl] = u_ub - u_bar[k, :]
        # lb[k * dim_ctrl:(k+1) * dim_ctrl] = u_lb - u_bar[k, :]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [
                    #   G @ x <= ub,
                    #   lb <= G @ x,
                      A @ x == b,
                      x[0:dim_state] == x0 - x_bar[0, :]
                    ])
    prob.solve(verbose=False, max_iter = 10000)
    
    return x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]

def Student_Controller_CMPC(x_bar, u_bar, x0, Fun_Jac_dt, param):
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = u_bar.shape[0] * u_bar.shape[1]
    n_x = x_bar.shape[0] * x_bar.shape[1]
    n_var = n_u + n_x

    n_eq  = x_bar.shape[1] * u_bar.shape[0] # dynamics
    n_ieq = u_bar.shape[1] * u_bar.shape[0] # input constraints
    
    # define the parameters
    Q = np.eye(4)  * 1
    R = np.eye(2)  * 10
    Pt = np.eye(4) * 1000
    
    # define the cost function
    np.random.seed(1)
    P = np.zeros((n_var, n_var))
    for k in range(u_bar.shape[0]):
        P[k * x_bar.shape[1]:(k+1) * x_bar.shape[1], k * x_bar.shape[1]:(k+1) * x_bar.shape[1]] = Q
        P[n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1], n_x + k * u_bar.shape[1]:n_x + (k+1) * u_bar.shape[1]] = R
    
    P[n_x - x_bar.shape[1]:n_x, n_x - x_bar.shape[1]:n_x] = Pt
    P = (P.T + P) / 2
    q = np.zeros((n_var, 1))
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    G = np.zeros((n_ieq, n_var))
    ub = np.zeros(n_ieq)
    lb = np.zeros(n_ieq)
    
    u_ub = np.array([  4,  0.8])
    u_lb = np.array([-10, -0.8])
    
    for k in range(u_bar.shape[0]):
        AB = Fun_Jac_dt(x_bar[k, :], u_bar[k, :], param)
        A[k * dim_state:(k+1) * dim_state,      k * dim_state:(k+1) * dim_state]       = AB[0] # AB[0:dim_state, 0:dim_state]
        A[k * dim_state:(k+1) * dim_state,  (k+1) * dim_state:(k+2) * dim_state]       = -np.eye(dim_state)
        A[k * dim_state:(k+1) * dim_state, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]  = AB[1] # AB[0:dim_state, dim_state:]
        
        G[k * dim_ctrl:(k+1) * dim_ctrl, n_x + k * dim_ctrl:n_x + (k+1) * dim_ctrl]    = np.eye(dim_ctrl)
        ub[k * dim_ctrl:(k+1) * dim_ctrl] = u_ub - u_bar[k, :]
        lb[k * dim_ctrl:(k+1) * dim_ctrl] = u_lb - u_bar[k, :]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [
                      G @ x <= ub,
                      lb <= G @ x,
                      A @ x == b,
                      x[0:dim_state] == x0 - x_bar[0, :]
                    ])
    prob.solve(verbose=False, max_iter = 10000)
    
    return x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]