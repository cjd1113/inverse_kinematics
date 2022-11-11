
import numpy as np
from matplotlib import pyplot as plt
from math_utils import body_angular_velocities, DCM_dot, Rx_rotation, Ry_rotation, Rz_rotation, Rxyz_rotation


if __name__ == '__main__':

    #Define a time vector
    t = np.linspace(0, 10, 100)

    # Generate Euler angles in the inertial frame
    phi = np.sin(2*t)
    theta = np.sin(3*t)
    psi = np.sin(4*t)
    # Compute Euler angle derivatives
    phi_dot = np.diff(phi)
    theta_dot = np.diff(theta)
    psi_dot = np.diff(psi)
    # Pre-allocate memory for body angular velocities
    wxb = np.zeros(len(t) - 2)
    wyb = np.zeros(len(t) - 2)
    wzb = np.zeros(len(t) - 2)

    # Generate inertial frame positions
    px = 2*(t**2)
    py = 0*t
    pz = 0*t

    # Generate inertial frame velocities
    vx = np.diff(px)
    vy = np.diff(py)
    vz = np.diff(pz)

    # Generate inertial frame accelerations
    ax = np.diff(vx)
    ay = np.diff(vy)
    az = np.diff(vz)

    ab = np.zeros([3, len(t) - 2])
    # axb = np.zeros(len(t) - 2)
    # ayb = np.zeros(len(t) - 2)
    # azb = np.zeros(len(t) - 2)

    for i in range(len(t)-2):

        Rxyz = Rxyz_rotation(Rx_rotation(phi[i]), Ry_rotation(theta[i]), Rz_rotation(psi[i]))
        dot_Rxyz = DCM_dot(phi[i], theta[i], psi[i], phi_dot[i], theta_dot[i], psi_dot[i])
        wxb_temp, wyb_temp, wzb_temp = body_angular_velocities(dot_Rxyz, Rxyz)
        wxb[i] = wxb_temp
        wyb[i] = wyb_temp
        wzb[i] = wzb_temp

        a_vec = np.array([ax[i], ay[i], az[i]])
        ab[:, i] = np.dot(Rxyz.T, a_vec)

    plt.figure()
    plt.plot(wxb)
    plt.plot(wyb)
    plt.plot(wzb)
    plt.legend(['wxb', 'wyb', 'wzb'])
    plt.title('Body Frame Angular Velocities')

    plt.figure()
    plt.plot(ab[0,:])
    plt.plot(ab[1,:])
    plt.plot(ab[2,:])
    plt.legend(['axb', 'ayb', 'azb'])
    plt.title('Body Frame Accelerations')
    plt.show()
