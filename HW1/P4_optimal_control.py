import numpy as np
import math
import scikits.bvp_solver
import matplotlib.pyplot as plt
from utils import *

dt = 0.005

def ode_fun(tau, z):
    """
    This function computes the dz given tau and z. It is used in the bvp solver.
    Inputs:
        tau: the independent variable. This must be the first argument.
        z: the state vector. The first three states are [x, y, th, ...]
    Output:
        dz: the state derivative vector. Returns a numpy array.
    """
    ########## Code starts here ##########
    # z = [x, y, th, p1, p2, p3, r]
    # idx  0, 1, 2,  3,  4,  5,  6
    V = - 0.5 * (z[3] * np.cos(z[2]) + z[4] * np.sin(z[2]))
    om = - 0.5 * z[5]
    xdot = z[6] * V * np.cos(z[2])
    ydot = z[6] * V * np.sin(z[2])
    thdot = z[6] * om
    p1dot = 0
    p2dot = 0
    p3dot = z[6] * V * (z[3] * np.sin(z[2]) - z[4] * np.cos(z[2]))
    rdot = 0
    dz = np.hstack((xdot, ydot, thdot, p1dot, p2dot, p3dot, rdot))
    ########## Code ends here ##########
    return dz


def bc_fun(za, zb):
    """
    This function computes boundary conditions. It is used in the bvp solver.
    Inputs:
        za: the state vector at the initial time
        zb: the state vector at the final time
    Output:
        bca: tuple of boundary conditions at initial time
        bcb: tuple of boundary conditions at final time
    """
    # final goal pose
    xf = 5
    yf = 5
    thf = -np.pi/2.0
    xf = [xf, yf, thf]
    # initial pose
    x0 = [0, 0, -np.pi/2.0]
    ########## Code starts here ##########
    # global lam defined in main()
    Hf = lam + 0.25 * (zb[3] * np.cos(zb[2]) + zb[4] * np.sin(zb[2])) ** 2 - \
        0.25 * zb[5]**2 + 0.5 * zb[4] * (zb[3] * np.cos(zb[2]) + zb[4] * np.sin(zb[2]))
    bca = np.array(za[0:3] - x0)
    bcb = np.array([zb[0] - xf[0], zb[1] - xf[1], zb[2] - xf[2], Hf])
    ########## Code ends here ##########
    return (bca, bcb)

def solve_bvp(problem_inputs, initial_guess):
    """
    This function solves the bvp_problem.
    Inputs:
        problem_inputs: a dictionary of the arguments needs to define the problem
                        num_ODE, num_parameters, num_left_boundary_conditions,
                        boundary_points, function, boundary_conditions
        initial_guess: initial guess of the solution
    Output:
        z: a numpy array of the solution. It is of size [time, state_dim]

    Read this documentation -- https://pythonhosted.org/scikits.bvp_solver/tutorial.html
    """
    problem = scikits.bvp_solver.ProblemDefinition(**problem_inputs)
    soln = scikits.bvp_solver.solve(problem, solution_guess=initial_guess)

    # Test if time is reversed in bvp_solver solution
    flip, tf = check_flip(soln(0))
    t = np.arange(0,tf,dt)
    z = soln(t/tf)
    if flip:
        z[3:7,:] = -z[3:7,:]
    z = z.T # solution arranged so that it is [time, state_dim]
    return z

def compute_controls(z):
    """
    This function computes the controls V, om, given the state z. It is used in main().
    Input:
        z: z is the state vector for multiple time instances. It has size [time, state_dim]
    Outputs:
        V: velocity control input
        om: angular rate control input
    """
    ########## Code starts here ##########
    N, dim = np.shape(z)
    V = np.zeros(N)
    om = np.zeros(N)
    for idx, row in enumerate(z):
        V[idx] = - 0.5 * (row[3] * np.cos(row[2]) + row[4] * np.sin(row[2]))
        om[idx] = - 0.5 * row[5]
    ########## Code ends here ##########

    return V, om

def main():
    """
    This function solves the specified bvp problem and returns the corresponding optimal control sequence
    Outputs:
        V: optimal V control sequence 
        om: optimal om control sequence
    You are required to define the problem inputs, initial guess, and compute the controls

    Hint: The total time is between 15-25
    """
    ########## Code starts here ##########
    global lam
    lam = 0.25
    num_ODE = 7
    num_parameters = 0
    num_left_boundary_conditions = 3
    boundary_points = (0,1)
    function = ode_fun
    boundary_conditions = bc_fun
    initial_guess = (0, 0, -np.pi/2, 0, 1.0, 1.0, 20)
    ########## Code ends here ##########

    problem_inputs = {
                      'num_ODE' : num_ODE,
                      'num_parameters' : num_parameters,
                      'num_left_boundary_conditions' : num_left_boundary_conditions,
                      'boundary_points' : boundary_points,
                      'function' : function,
                      'boundary_conditions' : boundary_conditions
                     }

    z = solve_bvp(problem_inputs, initial_guess)
    V, om = compute_controls(z)
    return z, V, om

if __name__ == '__main__':
    z, V, om = main()
    tf = z[0,-1]
    t = np.arange(0,tf,dt)
    x = z[:,0]
    y = z[:,1]
    th = z[:,2]
    data = {'z': z, 'V': V, 'om': om}
    save_dict(data, 'data/optimal_control.pkl')
    maybe_makedirs('plots')

    # plotting
    # plt.rc('font', weight='bold', size=16)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y,'k-',linewidth=2)
    plt.quiver(x[1:-1:200], y[1:-1:200],np.cos(th[1:-1:200]),np.sin(th[1:-1:200]))
    plt.grid(True)
    plt.plot(0,0,'go',markerfacecolor='green',markersize=15)
    plt.plot(5,5,'ro',markerfacecolor='red', markersize=15)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis([-1, 6, -1, 6])
    plt.title('Optimal Control Trajectory')

    plt.subplot(1, 2, 2)
    plt.plot(t, V, linewidth=2)
    plt.plot(t, om,linewidth=2)
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
    plt.title("Optimal control sequence, $t_f$ = {} s, $\lambda$ = {}".format(t[-1], lam))
    plt.tight_layout()
    plt.savefig('plots/optimal_control.pdf')
    plt.show()
