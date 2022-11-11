
import numpy as np
from matplotlib import pyplot as plt
from math_utils import body_angular_velocities, DCM_dot, Rx_rotation, Ry_rotation, Rz_rotation, Rxyz_rotation


if __name__ == '__main__':

    t = np.linspace(0, 10, 100)
    x = np.sin(2*t)
    dx = np.diff(x)
    plt.figure()
    plt.plot(t, x)
    plt.plot(t[1:], dx)
    plt.legend(['x','dx'])

    phi = 1
    theta = 2
    psi = 3
    phi_dot = 10
    theta_dot = 11
    psi_dot = 12

    Rxyz = Rxyz_rotation(Rx_rotation(phi), Ry_rotation(theta), Rz_rotation(psi))
    dot_Rxyz = DCM_dot(phi, theta, psi, phi_dot, theta_dot, psi_dot)
    wxb, wyb, wzb = body_angular_velocities(dot_Rxyz, Rxyz)

    print(Rxyz)
    plt.show()
