import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from typing import Any, Union, List, Optional


def apply_homography_to_image_points(height: int, width: int, homography: np.ndarray, keep_z: bool = True) -> np.ndarray:
    """
    Maps all pixel coordinates in an image using a homography matrix.

    Parameters:
    - height: image height
    - width: image width
    - homography: np.ndarray - 3x3 homography transformation matrix.

    Returns:
    - transformed_points: np.ndarray - Mapped 2D coordinates after homography transformation with shape (height, width, 2 or 3).
    """
    # Generate all (x, y) coordinate pairs for the image
    y_indices, x_indices = np.indices((height, width))
    all_points = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices).ravel()])  # [3, H*W]

    # Apply the homography matrix to all points
    transformed_points = (homography @ all_points)  # [3, H*W]

    # Normalize by the third row to convert from homogeneous to 2D coordinates
    transformed_points[:2] /= transformed_points[2]
    transformed_points = transformed_points.T.reshape(height, width, 3)

    # Keep or remove the z coordinate based on the keep_z flag
    if keep_z:
        return transformed_points
    else:
        return transformed_points[..., :2]


def generate_mask(transformed_points: Optional[np.ndarray], height: int, width: int, margin=0) -> np.ndarray:
    """
    Generates a mask indicating whether transformed points are within the image bounds.
    
    Parameters:
    - transformed_points: np.ndarray - Transformed 2D coordinates with shape (height, width, 2 or 3).
    - image_shape: tuple - Shape of the original image (height, width).
    
    Returns:
    - mask: np.ndarray - Binary mask of shape (height, width, 1), where 1 indicates in-bounds points and 0 indicates out-of-bounds.
    """
    # Extract coordinates from transformed_points
    if transformed_points.shape[-1] == 2:
        x_coords, y_coords = np.split(transformed_points, 2, axis=-1)
        z_coords = None
    else:
        x_coords, y_coords, z_coords = np.split(transformed_points, 3, axis=-1)

    # Initialize mask with zeros
    target_height, target_width = transformed_points.shape[:2]
    mask = np.ones((target_height, target_width, 1), dtype=np.uint8)

    # Create mask by checking if each point is within the image bounds
    mask &= (x_coords >= 0 - margin) & (x_coords <= width - 1 + margin)
    mask &= (y_coords >= 0 - margin) & (y_coords <= height - 1 + margin)

    # If z_coords is provided, also check if the depth is positive
    if z_coords is not None:
        mask &= (z_coords >= 0)

    # Convert boolean mask to an integer mask (0 or 1)
    return mask


def generate_K_R(fov, theta, phi, height, width, order='yx'):
    f = 0.5 * width / np.tan(np.radians(0.5 * fov))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    R_x, _ = cv2.Rodrigues(x_axis * np.radians(phi))
    R_y, _ = cv2.Rodrigues(y_axis * np.radians(theta))

    # Combine rotations in specified order
    if order == 'yx':
        R = R_y @ R_x
    else:
        R = R_x @ R_y
    
    return K, R


def pers2pers(image, fov, theta, phi, fov2, theta2, phi2, height2, width2, mask_input=False, return_mask=False, margin=0):
    """
    Args:
        image: np.ndarray, [H, W, C]
    """
    ori_ndim = image.ndim
    height, width = image.shape[:2]

    K1, R1 = generate_K_R(fov, theta, phi, height, width)
    K2, R2 = generate_K_R(fov2, theta2, phi2, height2, width2)

    R_relative = np.linalg.inv(R2) @ R1
    homo_matrix = K2 @ R_relative @ np.linalg.inv(K1)  # (3, 3)

    if mask_input:
        mask = generate_mask(apply_homography_to_image_points(height, width, homo_matrix), height2, width2, margin=margin)
        image_warp = cv2.warpPerspective(image * mask, homo_matrix, (width2, height2))
        if image_warp.ndim == 2 and ori_ndim == 3:  # cv2.warpPerspective seems to automatically reduce the axis with dimension 1
            image_warp = image_warp[:, :, None]
    else:
        mask = generate_mask(apply_homography_to_image_points(height2, width2, np.linalg.inv(homo_matrix)), height, width, margin=margin)
        image_warp = cv2.warpPerspective(image, homo_matrix, (width2, height2), flags=cv2.INTER_LINEAR)  # flags=cv2.INTER_CUBIC
        if image_warp.ndim == 2 and ori_ndim == 3:  # cv2.warpPerspective seems to automatically reduce the axis with dimension 1
            image_warp = image_warp[:, :, None]
        image_warp = image_warp * mask

    if return_mask:
        return image_warp, mask
    else:
        return image_warp


def npers2pers(images: List[np.ndarray], fovs, thetas, phis, fov2, theta2, phi2, height2, width2, overlap=False):
    dtype = images[0].dtype
    _, _, n_channels = images[0].shape
    image_combine = np.zeros((height2, width2, n_channels), dtype=np.float32)
    image_weight = np.zeros((height2, width2, n_channels), dtype=np.float32)
    margin = 0 if overlap else 1
    for image, fov, theta, phi in zip(images, fovs, thetas, phis):
        image_warp, mask = pers2pers(image, fov, theta, phi, fov2, theta2, phi2, height2, width2, return_mask=True, margin=margin)
        image_combine += image_warp.astype(np.float32)
        image_weight += mask.astype(np.float32)
    if overlap:
        image_combine /= image_weight
    if dtype == np.uint8:
        image_combine = image_combine.clip(0.0, 255.0)
    return image_combine.astype(dtype)


def map_pers_coords(wfov, theta, phi, h, w, output_type:str='xyz', normalize:bool=False):
    hfov = float(h) / w * wfov

    w_len = np.tan(np.radians(wfov / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x_map = np.ones([h, w], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, w), [h, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, h), [w, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / \
        np.repeat(D[:, :, np.newaxis], 3, axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    xyz = xyz.reshape([h * w, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    
    xyz = rearrange(xyz, '(h w) c -> c h w', h=h, w=w)

    if output_type == 'xyz':
        return xyz
    
    elif output_type == 'lonlat':
        x, y, z = xyz
        lon = np.arctan2(y, x)
        lat = np.arcsin(z)
        coords = np.stack((lon, lat), axis=0)
        if normalize:
            coords /= np.pi
        return coords

    elif output_type == 'lonlat_loop':
        x, y, z = xyz
        x = np.abs(x)
        y = np.abs(y)
        lon = np.arctan2(y, x)
        lat = np.arcsin(z)
        coords = np.stack((lon, lat), axis=0)
        if normalize:
            coords /= (np.pi * 0.5)
        return coords

    elif output_type == 'uv':
        raise NotImplementedError()
        x, y, z = xyz
        u = np.arctan2(x, z)
        v = np.arctan2(y, np.sqrt(x**2 + z**2))
        return np.stack((u, v), axis=0)
    
    else:
        raise ValueError(f'Unknown output_type: {output_type}')


def prepare_positions(
    height: int,
    width: int,
    cameras: dict,
    dtype: torch.dtype,
    device: torch.device,
    output_type:str='xyz',
    normalize:bool=False,
):
    coords = []
    for fov, theta, phi in zip(cameras['fov'][0], cameras['theta'][0], cameras['phi'][0]):
        fov, theta, phi = fov.item(), theta.item(), phi.item()
        coord = map_pers_coords(fov, theta, phi, height, width, output_type=output_type, normalize=normalize)
        coords.append(torch.tensor(coord, dtype=dtype, device=device))
    coords = torch.stack(coords, dim=0)  # [M, C, H, W]
    return coords
