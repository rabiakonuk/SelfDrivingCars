def Student_Controller(t, x, param):
    import numpy as np
    vd = param["vd"]
    v0 = param["v0"]

    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]
    
    lam = 10.0
    alpha = 0.19
    
    B    = x[0] - 1.8 * x[1] - 0.5 * (x[1] - v0)**2 / Cdg
    dBdt = v0 - x[1]
    
    P = np.diag([1.0, 1e7])
    
    A = np.zeros([5, 2])
    A[0, 0] = (x[1] - vd) / m
    A[0, 1] = -1.
    A[1,0] = (1.8 + (x[1] - v0)/Cdg ) / m
    A[2,0] =  1.
    A[3,0] = -1.
    A[4,1] = -1.

    b = np.zeros([5])
    b[0] = - lam * (x[1] - vd)**2 / 2
    b[1] = alpha * B + dBdt
    b[2] = Cag * m
    b[3] = Cdg * m
    
    q = np.zeros([2, 1])
    
    return A, b, P, q