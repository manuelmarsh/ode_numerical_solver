# -*- coding: utf-8 -*-
"""
@author: Manuel Martini
"""
import numpy as np  
import matplotlib.pyplot as plt

def ode_solver(f, t, x0, method, error=10**(-7)):
    """
    Solves the ordinary differential equation (ODE) using specified methods.

    Parameters:
    f: function representing the derivative of x with respect to t
    t: discretization of the time variable
    x0: initial state
    method: callable representing the approximation method used. 
            It has an attribute 'steps' indicating the number of steps used by the method.
            Alternatively, method can be a string with one of three methods:
            ['Eulero_forward', 'Eulero_backward', 'Crank_Nicholson']
    error: tolerance for convergence (default: 1e-7)
    
    Returns:
    x: array of computed values at each time step
    """
    
    # Default methods for different ODE solving techniques
    default_methods = {
        'Eulero_forward': lambda f, t, x, i: f(t[i], x[i]),
        'Eulero_backward': lambda f, t, x, i: f(t[i + 1], x[i + 1]),
        'Crank_Nicholson': lambda f, t, x, i: 0.5 * (f(t[i], x[i]) + f(t[i + 1], x[i + 1]))
    }
    
    # Determine if the provided method is a string or callable function
    if method in default_methods:  # if method is a string
        method = default_methods[method]
        num_steps = 1  # Assign 1 step for consistency in indexing
    else:
        num_steps = method.steps  # if method is a callable with steps attribute
    
    n = t.size  # Number of time steps
    x = np.zeros((n,) + x0.shape)  # Initialize solution array
    x[0] = x0  # Set initial condition
    
    # First set of calculations using a single-step method (e.g., for Crank-Nicholson)
    for i in range(num_steps - 1):  # Calculate initial values in x
        h = t[i + 1] - t[i]  # Time step size
        x[i + 1] = x[i] + h * f(t[i], x[i])  # Forward Euler method
        x_next = x[i] + h * 0.5 * (f(t[i], x[i]) + f(t[i + 1], x[i + 1]))
        
        # Iteratively refine x[i + 1] until convergence
        while np.sum(np.abs(x_next - x[i + 1])) > error:
            x[i + 1] = x_next
            x_next = x[i] + h * 0.5 * (f(t[i], x[i]) + f(t[i + 1], x[i + 1]))
        x[i + 1] = x_next  # Store converged value

    # Second set of calculations for remaining time steps
    for i in range(num_steps - 1, n - 1):
        h = t[i + 1] - t[i]
        x[i + 1] = x[i] + h * f(t[i], x[i])  # Initial guess using Forward Euler
        x_next = x[i] + h * (method(f, t, x, i))  # Compute using chosen method
        
        # Iteratively refine x[i + 1] until convergence
        while np.sum(np.abs(x_next - x[i + 1])) > error:
            x[i + 1] = x_next
            x_next = x[i] + h * (method(f, t, x, i))
        x[i + 1] = x_next  # Store converged value

    return x  # Return the array of computed values

if __name__ == '__main__':
    class Adams_Moulton:
        """
        Implementation of the Adams-Moulton method for ODE solving.

        Attributes:
        steps: number of steps used by the method
        """
        
        def __init__(self):
            self.steps = 3  # Number of steps for the Adams-Moulton method
        
        def __call__(self, f, t, x, i):
            """
            Calculate the value using the Adams-Moulton method.

            Parameters:
            f: function representing the derivative
            t: time array
            x: array of computed values
            i: current index
            
            Returns:
            computed value at time step i
            """
            # Adams-Moulton formula combining the function evaluations
            return (9 / 24 * f(t[i + 1], x[i + 1]) +
                    19 / 24 * f(t[i], x[i]) -
                    5 / 24 * f(t[i - 1], x[i - 1]) +
                    1 / 24 * f(t[i - 2], x[i - 2]))
    
    # Plotting section
    fig = plt.figure(dpi=200)  # Create a figure for plotting
    f = lambda t, u: -t + 1  # Define the ODE function
    time = np.linspace(0, 10, 100)  # Time array from 0 to 10
    x0 = np.array(0)  # Initial state
    method = Adams_Moulton()  # Choose the Adams-Moulton method
    sequence = ode_solver(f, time, x0, method)  # Solve the ODE
    
    # Print computed sequence and plot results
    print(sequence)
    plt.scatter(time, sequence, label='Adams Moulton', s=5)  # Plot computed values
    plt.scatter(time, np.array(-0.5 * time**2 + time), label='Exact solution', s=1)  # Plot exact solution
    plt.legend()  # Show legend
    plt.grid()  # Add grid to plot
    plt.title('ODE Approximations')  # Set plot title
    plt.xlabel('t')  # Set x-axis label
    plt.ylabel('x')  # Set y-axis label
    plt.show()  # Display the plot

   
