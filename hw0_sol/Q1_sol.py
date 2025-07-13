def linear_ode(A):
    import numpy as np
    from scipy.linalg import expm

    # Define the system matrix A

    # Define the time step and simulation duration
    dt = 0.01  # Time step
    T = 6.0    # Total simulation time
    num_steps = int(T / dt)

    # Initial state
    x0 = np.array([1, 1])

    ### your code starts here ###

    # Compute the matrix exponential of A
    exp_A = expm(A * dt)

    # Simulate the system over time
    state_history = [x0]
    for _ in range(num_steps):
        x_next = np.dot(exp_A, state_history[-1])
        state_history.append(x_next)

    # Extract x and y coordinates for the phase portrait
    x1_vals = [x[0] for x in state_history]
    x2_vals = [x[1] for x in state_history]

    # Compute continuous-time eigenvalues
    continuous_eigenvalues, _ = np.linalg.eig(A)

    # Compute discretized eigenvalues
    discretized_eigenvalues, _ = np.linalg.eig(exp_A)

    ### your code ends here ###
    return continuous_eigenvalues, discretized_eigenvalues, exp_A, x1_vals, x2_vals