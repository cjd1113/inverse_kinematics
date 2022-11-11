import numpy as np


def quaternionKinematicsdt(wx, wy, wz, q, dt) -> np.ndarray:

    q_next = np.dot(0.5*dt*np.array([[0, -wx, -wy, -wz],
                                     [wx, 0, wz, -wy],
                                     [wy, -wz, 0, wx],
                                     [wz, wy, -wx, 0]]) + np.eye(4), q)

    return q_next / np.linalg.norm(q_next)

