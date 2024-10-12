# ODE Numerical Solver

This project provides an implementation of various methods to solve ordinary differential equations (ODEs). It includes a general ODE solver function that can handle different numerical methods, including Forward Euler, Backward Euler, and Crank-Nicholson methods. The project also features the Adams-Moulton method as an example of a multi-step method.

## Features

- **Flexible ODE Solver**: A customizable function that accepts different methods for solving ODEs.
- **Multiple Methods**: Internal implementations of various numerical methods:
  - Forward Euler
  - Backward Euler
  - Crank-Nicholson
- **Error Handling**: Built-in error tolerance for convergence in the solution process.
- **Visualization**: Plots the computed solution alongside the exact solution for comparison.

## Requirements

To run this project, ensure you have the following libraries installed:

- Python
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install numpy matplotlib
```
## Usage
Define the ODE: Create a function that represents the derivative of the variable with respect to time.
Choose a Time Array: Define the time points at which you want to compute the solution.
Set Initial Conditions: Define the initial state of your system.
Select a Method: Choose one of the available methods to solve the ODE.
Call the Solver: Use the ode_solver function with the defined parameters.
