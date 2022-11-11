import numpy as np
import scipy as scipy


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
             np.cos(psi_t) - psi_d_t * np.sin(phi_t) * np.sin(theta_t) * np.sin(psi_t) + phi_d_t * np.sin(
        phi_t) * np.sin(psi_t) \
             - psi_d_t * np.cos(phi_t) * np.cos(psi_t)
    C22dot = phi_d_t * np.cos(phi_t) * np.sin(theta_t) * np.sin(psi_t) + theta_d_t * np.sin(phi_t) * np.cos(
        theta_t) * np.sin(psi_t) \
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


def euler_to_quats(phi, theta, psi) -> np.ndarray:
    sin_psi = np.sin(psi / 2)
    cos_psi = np.cos(psi / 2)
    sin_phi = np.sin(phi / 2)
    cos_phi = np.cos(phi / 2)
    sin_theta = np.sin(theta / 2)
    cos_theta = np.cos(theta / 2)

    # The following equations represent the 123 rotation, per equation 297 of Diebel2006.
    q0_init = cos_psi * cos_theta * cos_phi + sin_psi * sin_theta * sin_phi
    q1_init = cos_psi * cos_theta * sin_phi - sin_psi * sin_theta * cos_phi
    q2_init = cos_psi * sin_theta * cos_phi + sin_psi * cos_theta * sin_phi
    q3_init = sin_psi * cos_theta * cos_phi - cos_psi * sin_theta * sin_phi

    q = np.array([q0_init, q1_init, q2_init, q3_init])

    return q


def quat_to_euler(q) -> tuple:
    R = np.zeros([3, 3])

    R[0, 0] = 2 * q[0] ** 2 - 1 + 2 * q[1] ** 2
    R[1, 0] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[2, 0] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[2, 1] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[2, 2] = 2 * q(0) ** 2 - 1 + 2 * q(3) ** 2

    phi = np.atan2(R[2, 1], R[2, 2])
    theta = -np.atan(R[2, 0] / np.sqrt(1 - R[2, 0] ** 2))
    psi = np.atan2(R[1, 0], R[0, 0])

    return phi, theta, psi


def quat_to_rot_mx(q) -> np.ndarray:
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # Inertial to Body
    R_q_rot_ItoB = np.array(
        [[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
         [2 * q1 * q2 - 2 * q0 * q3, q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * q2 * q3 + 2 * q0 * q1],
         [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]])

    # Body to Inertial
    R_q_rot_BtoI = R_q_rot_ItoB.T

    return R_q_rot_BtoI
