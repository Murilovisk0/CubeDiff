import numpy as np
import cv2


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def random_sample_spherical(n):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz)
    return xyz


def random_sample_camera(n):
    xyz = random_sample_spherical(n)
    phi = np.arcsin(xyz[:, 2].clip(-1, 1))
    theta = np.arctan2(xyz[:, 0], xyz[:, 1])
    return theta, phi


def horizon_sample_camera(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi = np.zeros_like(theta)
    return theta, phi


def icosahedron_sample_camera():
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)
    theta_step = 2.0 * np.pi / 5.0

    thetas = []
    phis = []
    for triangle_index in range(20):
        # 1) the up 5 triangles
        if 0 <= triangle_index <= 4:
            theta = - np.pi + theta_step / 2.0 + triangle_index * theta_step
            phi = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

        # 2) the middle 10 triangles
        # 2-0) middle-up triangles
        if 5 <= triangle_index <= 9:
            triangle_index_temp = triangle_index - 5
            theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            phi = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)

        # 2-1) the middle-down triangles
        if 10 <= triangle_index <= 14:
            triangle_index_temp = triangle_index - 10
            theta = - np.pi + triangle_index_temp * theta_step
            phi = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

        # 3) the down 5 triangles
        if 15 <= triangle_index <= 19:
            triangle_index_temp = triangle_index - 15
            theta = - np.pi + triangle_index_temp * theta_step
            phi = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

        thetas.append(theta)
        phis.append(phi)

    return np.array(thetas), np.array(phis)


def skybox_sample_camera(skybox_order: str = 'equilib', extend: bool = False, degree: bool = False):
    theta, phi = np.zeros((6,)), np.zeros((6,))
    if skybox_order == 'equilib':
        # L, F, R, B, U, D
        theta[:4] = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        phi[-2] = np.pi / 2
        phi[-1] = -np.pi / 2
    elif skybox_order == 'mp3d':
        # U, L, F, R, B, D
        theta[[2, 3, 4, 1]] = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        phi[0] = np.pi / 2
        phi[-1] = -np.pi / 2
    else:
        raise ValueError('Unknown skybox order: {}'.format(skybox_order))
    if extend:
        theta_ext, phi_ext = np.zeros((8,)), np.zeros((8,))
        for i in range(0, 4):
            theta_ext[i] = theta_ext[i+4] = theta[i] + np.pi / 4
            phi_ext[i] = np.pi / 4
            phi_ext[i+4] = - np.pi / 4
        theta = np.concatenate([theta, theta_ext])
        phi = np.concatenate([phi, phi_ext])
    if degree:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return theta, phi


def get_K(fov, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)


def get_R(theta, phi):
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    return R2 @ R1


def get_K_R(fov, theta, phi, height, width):
    K = get_K(fov=fov, height=height, width=width)
    R = get_R(theta=theta, phi=phi)
    return K, R
