from numpy import np
from scipy.linalg import eig


def get_rotation_matrix(pos: np.ndarray | list, vel: np.ndarray | list, mass: np.ndarray | list):
    """
    Computes a rotation matrix based on the z-direction of the angular momentum vector of chosen
    particles. The rotation is chosen such that the bulk angular momentum points in the z-direction

    Parameters
    ----------
    pos : np.array-like with shape (N, 3)
        The positions of the particles used to define the rotation matrix.
    vel : np.array-like with shape (N, 3)
        The velocities of the particles used to define the rotation matrix.
    mass : np.array-like with shape (N,)
        The masses of the particles used to define the rotation matrix.

    Returns
    -------
    rotation_matrix : np.array with shape (3,3)
    The rotation matrix that maximizes the z-direction of the bulk angular momentum.
    """
    all_l = np.linalg.cross(pos, vel)
    totl = all_l.sum(axis=0)
    ldir = totl / np.sqrt((totl**2).sum())
    tensor = np.zeros((3, 3))

    px = pos[:, 0]
    py = pos[:, 1]
    pz = pos[:, 2]

    tensor[0, 0] = (mass * (py * py + pz * pz)).sum()
    tensor[1, 1] = (mass * (px * px + pz * pz)).sum()
    tensor[2, 2] = (mass * (px * px + py * py)).sum()

    tensor[0, 1] = -(mass * px * py).sum()
    tensor[1, 0] = tensor[0, 1]
    tensor[0, 2] = -(mass * px * pz).sum()
    tensor[2, 0] = tensor[0, 2]
    tensor[1, 2] = -(mass * py * pz).sum()
    tensor[2, 1] = tensor[1, 2]

    eigval, eigvec = eig(tensor)

    vector1 = (ldir * eigvec[:, 0]).sum()
    vector2 = (ldir * eigvec[:, 1]).sum()
    vector3 = (ldir * eigvec[:, 2]).sum()

    matrix_a = np.abs(np.array([vector1, vector2, vector3]))
    (i,) = np.where(matrix_a == matrix_a.max())
    xdir = eigvec[:, i[0]]

    if (xdir * ldir).sum() < 0:
        xdir *= -1.0

    (j,) = np.where(matrix_a != matrix_a.max())
    i2 = eigval[j].argsort()
    ydir = eigvec[:, j[i2[1]]]

    if ydir[0] < 0:
        ydir *= -1.0

    zdir = np.cross(xdir, ydir)

    rotation_matrix = np.array([xdir, ydir, zdir]).transpose()
    return rotation_matrix
