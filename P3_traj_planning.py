import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate as itp
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        if t < self.t_before_switch:
            return self.traj_controller.compute_control(x,y,th,t)
        else:
            return self.pose_controller.compute_control(x,y,th,t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)
    x_old = path[:,0]
    y_old = path[:,1]
    n = np.shape(x_old)[0]
    # assert np.shape(y_old) == np.shape(x_old)
    t = np.zeros((n,))
    t_old = np.zeros((n,))
    # assert np.shape(x_old[1:] - x_old[:-1]) == (n-1,)
    t[1:] = np.linalg.norm(np.column_stack((x_old[1:] - x_old[:-1], y_old[1:]-y_old[:-1])), axis = 1) / V_des
    for i in range(n):
        t_old[i] = np.sum(t[:i+1])
    tck_x = itp.splrep(t_old, x_old, s = alpha)
    tck_y = itp.splrep(t_old, y_old, s = alpha)
    t_smoothed = np.arange(0.0, max(t_old), dt)
    n_new = np.shape(t_smoothed)[0]
    traj_smoothed = np.zeros((n_new,7))
    traj_smoothed[:, 0] = itp.splev(t_smoothed, tck_x)
    traj_smoothed[:, 1] = itp.splev(t_smoothed, tck_y)
    traj_smoothed[:, 3] = itp.splev(t_smoothed, tck_x, der = 1)
    traj_smoothed[:, 5] = itp.splev(t_smoothed, tck_x, der = 2)
    traj_smoothed[:, 4] = itp.splev(t_smoothed, tck_y, der = 1)
    traj_smoothed[:, 6] = itp.splev(t_smoothed, tck_y, der = 2)
    traj_smoothed[:, 2] = np.arctan2(traj_smoothed[:,4], traj_smoothed[:,3])
    ########## Code ends here ##########
    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(traj[-1,0], traj[-1,1], np.sqrt(traj[-1,3]**2+traj[-1,4]**2), np.arctan2(traj[-1,4], traj[-1,3]))
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
