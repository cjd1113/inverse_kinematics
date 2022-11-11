
import numpy as np
import scipy as scipy

def numerical_derivative(dt) -> float:



    return 10


def Rx_rotation(angle) -> np.ndarray:
    return np.array([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])


def Ry_rotation(angle) -> np.ndarray:
    return np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])


def Rz_rotation(angle) -> np.ndarray:
    return np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def Rxyz_rotation(Rx, Ry, Rz) -> np.ndarray:
    return np.dot(np.dot(Rx, Ry), Rz)


def DCM_dot(phi, theta, psi, phi_dot, theta_dot, psi_dot) -> np.ndarray:

    # Note that this is the derivative of the Z-Y-X (123) rotation matrix

    phi_t = phi
    theta_t = theta
    psi_t = psi
    phi_d_t = phi_dot
    theta_d_t = theta_dot
    psi_d_t = psi_dot

    C11dot = -theta_d_t * np.sin(theta_t) * np.cos(psi_t) - psi_d_t * np.sin(psi_t) * np.cos(theta_t)
    C12dot = -theta_d_t * np.sin(theta_t) * np.sin(psi_t) + psi_d_t * np.cos(theta_t) * np.cos(psi_t)
    C13dot = -theta_d_t * np.cos(theta_t)

    C21dot = phi_d_t * np.cos(phi_t) * np.sin(theta_t) * np.cos(psi_t) + theta_d_t * np.sin(phi_t) * np.cos(theta_t) * \
             np.cos(psi_t) - psi_d_t * np.sin(phi_t) * np.sin(theta_t) * np.sin(psi_t) + phi_d_t * np.sin(phi_t) * np.sin(psi_t) \
             - psi_d_t * np.cos(phi_t) * np.cos(psi_t)
    C22dot = phi_d_t * np.cos(phi_t) * np.sin(theta_t) * np.sin(psi_t) + theta_d_t * np.sin(phi_t) * np.cos(theta_t) * np.sin(psi_t) \
             + psi_d_t * np.sin(phi_t) * np.sin(theta_t) * np.cos(psi_t) - phi_d_t * np.sin(phi_t) * np.cos(psi_t) - \
             psi_d_t * np.cos(phi_t) * np.sin(psi_t)
    C23dot = phi_d_t * np.cos(phi_t) * np.cos(theta_t) - theta_d_t * np.sin(phi_t) * np.sin(theta_t)

    C31dot = -phi_d_t * np.sin(phi_t) * np.sin(theta_t) * np.cos(psi_t) + theta_d_t * np.cos(phi_t) * \
             np.cos(theta_t) * np.cos(psi_t) - psi_d_t * np.cos(phi_t) * np.sin(theta_t) * np.sin(psi_t) + \
             phi_d_t * np.cos(phi_t) * np.sin(psi_t) + psi_d_t * np.sin(phi_t) * np.cos(psi_t)
    C32dot = -phi_d_t * np.sin(phi_t) * np.sin(theta_t) * np.sin(psi_t) + theta_d_t * np.cos(phi_t) * \
             np.cos(theta_t) * np.sin(psi_t) + psi_d_t * np.cos(phi_t) * np.sin(theta_t) * np.cos(psi_t) - \
             phi_d_t * np.cos(phi_t) * np.cos(psi_t) + psi_d_t * np.sin(phi_t) * np.sin(psi_t)
    C33dot = -phi_d_t * np.sin(phi_t) * np.cos(theta_t) - theta_d_t * np.cos(phi_t) * np.sin(theta_t)

    return np.array([[C11dot, C12dot, C13dot], [C21dot, C22dot, C23dot], [C31dot, C32dot, C33dot]])


def body_angular_velocities(ddt_DCM, DCM) -> np.ndarray:

    # Unpack Czyx
    C11 = DCM[0, 0]
    C12 = DCM[0, 1]
    C13 = DCM[0, 2]
    C21 = DCM[1, 0]
    C22 = DCM[1, 1]
    C23 = DCM[1, 2]
    C31 = DCM[2, 0]
    C32 = DCM[2, 1]
    C33 = DCM[2, 2]

    # Unpack Czyxdot
    C11dot = ddt_DCM[0, 0]
    C12dot = ddt_DCM[0, 1]
    C13dot = ddt_DCM[0, 2]
    C21dot = ddt_DCM[1, 0]
    C22dot = ddt_DCM[1, 1]
    C23dot = ddt_DCM[1, 2]
    C31dot = ddt_DCM[2, 0]
    C32dot = ddt_DCM[2, 1]
    C33dot = ddt_DCM[2, 2]

    wxb = C21dot * C31 + C22dot * C32 + C23dot * C33
    wyb = C31dot * C11 + C32dot * C12 + C33dot * C13
    wzb = C11dot * C21 + C12dot * C22 + C13dot * C23

    return np.array([wxb, wyb, wzb])




