import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Union, List, Optional

from external import py360convert


def pad_equi(pano: torch.Tensor, padding: int):
    if padding <= 0:
        return pano

    if pano.ndim == 5:
        b, m = pano.shape[:2]
        pano_pad = rearrange(pano, 'b m c h w -> (b m c) h w')
    elif pano.ndim == 4:
        b = pano.shape[0]
        pano_pad = rearrange(pano, 'b c h w -> (b c) h w')
    else:
        raise NotImplementedError('pano should be 4 or 5 dim')

    pano_pad = F.pad(pano_pad, [padding, ] * 2, mode='circular')
    # pano_pad = torch.cat([
    #     pano_pad[..., :padding, :].flip((-1, -2)),
    #     pano_pad,
    #     pano_pad[..., -padding:, :].flip((-1, -2)),
    # ], dim=-2)

    if pano.ndim == 5:
        pano_pad = rearrange(pano_pad, '(b m c) h w -> b m c h w', b=b, m=m)
    elif pano.ndim == 4:
        pano_pad = rearrange(pano_pad, '(b c) h w -> b c h w', b=b)

    return pano_pad


def unpad_equi(pano_pad: torch.Tensor, padding: int):
    if padding <= 0:
        return pano_pad
    return pano_pad[..., padding:-padding]


class Equirectangular:
    def __init__(self, equirectangular):
        self.equirectangular = equirectangular

    @classmethod
    def from_file(cls, img_path):
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert("RGB")
        return cls(np.array(img))

    def to_cubemap(self, face_w=256, mode='bilinear'):
        from utils.cube import Cubemap
        return Cubemap(py360convert.e2c(self.equirectangular, face_w, mode, cube_format='horizon'), 'horizon')

    def to_perspective(self, fov, yaw, pitch, hw, mode='bilinear'):
        return py360convert.e2p(self.equirectangular, fov, yaw, pitch, hw, mode=mode)

    def rotate(self, degree):
        if round(degree) % 360 == 0:
            return
        shift = int(degree / 360 * self.equirectangular.shape[1])
        # if shift > 0, move to right, else move to left
        self.equirectangular = np.roll(self.equirectangular, -shift, axis=1)

    def flip(self, flip=True):
        if flip:
            self.equirectangular = np.flip(self.equirectangular, 1)

    def to_PIL_image(self) -> Image.Image:
        return Image.fromarray(self.equirectangular.astype(np.uint8))

    def show(self):
        self.to_PIL_image().show()

    def save(self, path):
        self.to_PIL_image().save(path)


if __name__ == '__main__':
    import argparse
    from utils.camera import icosahedron_sample_camera, horizon_sample_camera, random_sample_camera

    parser = argparse.ArgumentParser(description="Stitch Matterport3D Skybox")
    parser.add_argument("--mp3d_skybox_path", required=True,
                        help="Matterport3D mp3d_skybox path")
    parser.add_argument("--label_data_path", required=True,
                        help="Matterport3DLayoutAnnotation label_data path")
    parser.add_argument("--scene", default=None,
                        help="scene id", type=str)
    parser.add_argument("--view", default=None,
                        help="view id", type=str)
    args = parser.parse_args()

    cubemap = Cubemap.from_mp3d_skybox(args.mp3d_skybox_path, args.scene, args.view)
    equirectangular = cubemap.to_equirectangular(1024, 2048)
    equirectangular.save(os.path.join('debug', 'equirectangular.jpg'))

    theta, phi = icosahedron_sample_camera()
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"icosahedron_{i}.jpg"))

    theta, phi = horizon_sample_camera(8)
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"horizon_{i}.jpg"))

    theta, phi = random_sample_camera(20)
    theta, phi = np.rad2deg(theta), np.rad2deg(phi)
    for i, (t, p) in enumerate(zip(theta, phi)):
        perspective = equirectangular.to_perspective((90, 90), t, p, (512, 512))
        Image.fromarray(perspective.astype(np.uint8)).save(os.path.join('debug', f"random_{i}.jpg"))

    layout_json = os.path.join(args.label_data_path, f"{args.scene}_{args.view}_label.json")
    layout = Layout.from_json(layout_json)
    layout.render_to_files('debug')
