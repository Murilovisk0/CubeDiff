import torch
import torch.nn.functional as F
import numpy as np
import PIL
from PIL import Image
from einops import rearrange, repeat
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor


PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]


class DepthVisualizer:
    def __init__(
        self,
        do_normalize: bool = True,
        do_range_check: bool = True,
    ):
        super().__init__()

    @staticmethod
    def expand_tensor_or_array(images: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Expand a tensor or array to a specified number of images.
        """
        if isinstance(images, np.ndarray):
            if images.ndim == 2:  # [H,W] -> [1,H,W,1]
                images = images[None, ..., None]
            if images.ndim == 3:  # [H,W,C] -> [1,H,W,C]
                images = images[None]
        elif isinstance(images, torch.Tensor):
            if images.ndim == 2:  # [H,W] -> [1,1,H,W]
                images = images[None, None]
            elif images.ndim == 3:  # [1,H,W] -> [1,1,H,W]
                images = images[None]
        else:
            raise ValueError(f"Unexpected input type: {type(images)}")
        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image to a PyTorch tensor.
        """
        if np.issubdtype(images.dtype, np.integer) and not np.issubdtype(images.dtype, np.unsignedinteger):
            raise ValueError(f"Input image dtype={images.dtype} cannot be a signed integer.")
        if np.issubdtype(images.dtype, np.complexfloating):
            raise ValueError(f"Input image dtype={images.dtype} cannot be complex.")
        if np.issubdtype(images.dtype, bool):
            raise ValueError(f"Input image dtype={images.dtype} cannot be boolean.")

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def resize_antialias(
        image: torch.Tensor, size: Tuple[int, int], mode: str, is_aa: Optional[bool] = None
    ) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        antialias = is_aa and mode in ("bilinear", "bicubic")
        image = F.interpolate(image, size, mode=mode, antialias=antialias)

        return image

    @staticmethod
    def resize_to_max_edge(image: torch.Tensor, max_edge_sz: int, mode: str) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        max_orig = max(h, w)
        new_h = h * max_edge_sz // max_orig
        new_w = w * max_edge_sz // max_orig

        if new_h == 0 or new_w == 0:
            raise ValueError(f"Extreme aspect ratio of the input image: [{w} x {h}]")

        image = MarigoldImageProcessor.resize_antialias(image, (new_h, new_w), mode, is_aa=True)

        return image

    @staticmethod
    def pad_image(image: torch.Tensor, align: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        h, w = image.shape[-2:]
        ph, pw = -h % align, -w % align

        image = F.pad(image, (0, pw, 0, ph), mode="replicate")

        return image, (ph, pw)

    @staticmethod
    def unpad_image(image: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.dim() != 4:
            raise ValueError(f"Invalid input dimensions; shape={image.shape}.")

        ph, pw = padding
        uh = None if ph == 0 else -ph
        uw = None if pw == 0 else -pw

        image = image[:, :, :uh, :uw]

        return image

    @staticmethod
    def load_image_canonical(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, int]:
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_dtype_max = None
        if isinstance(image, (np.ndarray, torch.Tensor)):
            image = MarigoldImageProcessor.expand_tensor_or_array(image)
            if image.ndim != 4:
                raise ValueError("Input image is not 2-, 3-, or 4-dimensional.")
        if isinstance(image, np.ndarray):
            if np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.unsignedinteger):
                raise ValueError(f"Input image dtype={image.dtype} cannot be a signed integer.")
            if np.issubdtype(image.dtype, np.complexfloating):
                raise ValueError(f"Input image dtype={image.dtype} cannot be complex.")
            if np.issubdtype(image.dtype, bool):
                raise ValueError(f"Input image dtype={image.dtype} cannot be boolean.")
            if np.issubdtype(image.dtype, np.unsignedinteger):
                image_dtype_max = np.iinfo(image.dtype).max
                image = image.astype(np.float32)  # because torch does not have unsigned dtypes beyond torch.uint8
            image = MarigoldImageProcessor.numpy_to_pt(image)

        if torch.is_tensor(image) and not torch.is_floating_point(image) and image_dtype_max is None:
            if image.dtype != torch.uint8:
                raise ValueError(f"Image dtype={image.dtype} is not supported.")
            image_dtype_max = 255

        if not torch.is_tensor(image):
            raise ValueError(f"Input type unsupported: {type(image)}.")

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # [N,1,H,W] -> [N,3,H,W]
        if image.shape[1] != 3:
            raise ValueError(f"Input image is not 1- or 3-channel: {image.shape}.")

        image = image.to(device=device, dtype=dtype)

        if image_dtype_max is not None:
            image = image / image_dtype_max

        return image

    @staticmethod
    def check_image_values_range(image: torch.Tensor) -> None:
        if not torch.is_tensor(image):
            raise ValueError(f"Invalid input type={type(image)}.")
        if not torch.is_floating_point(image):
            raise ValueError(f"Invalid input dtype={image.dtype}.")
        if image.min().item() < 0.0 or image.max().item() > 1.0:
            raise ValueError("Input image data is partially outside of the [0,1] range.")

    def preprocess(
        self,
        image: PipelineImageInput,
        processing_resolution: Optional[int] = None,
        resample_method_input: str = "bilinear",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        if isinstance(image, list):
            images = None
            for i, img in enumerate(image):
                img = self.load_image_canonical(img, device, dtype)  # [N,3,H,W]
                if images is None:
                    images = img
                else:
                    if images.shape[2:] != img.shape[2:]:
                        raise ValueError(
                            f"Input image[{i}] has incompatible dimensions {img.shape[2:]} with the previous images "
                            f"{images.shape[2:]}"
                        )
                    images = torch.cat((images, img), dim=0)
            image = images
            del images
        else:
            image = self.load_image_canonical(image, device, dtype)  # [N,3,H,W]

        original_resolution = image.shape[2:]

        if self.config.do_range_check:
            self.check_image_values_range(image)

        if self.config.do_normalize:
            image = image * 2.0 - 1.0

        if processing_resolution is not None and processing_resolution > 0:
            image = self.resize_to_max_edge(image, processing_resolution, resample_method_input)  # [N,3,PH,PW]

        image, padding = self.pad_image(image, self.config.vae_scale_factor)  # [N,3,PPH,PPW]

        return image, padding, original_resolution

    @staticmethod
    def colormap(
        image: Union[np.ndarray, torch.Tensor],
        cmap: str = "Spectral",
        bytes: bool = False,
        _force_method: Optional[str] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Converts a monochrome image into an RGB image by applying the specified colormap. This function mimics the
        behavior of matplotlib.colormaps, but allows the user to use the most discriminative color maps ("Spectral",
        "binary") without having to install or import matplotlib. For all other cases, the function will attempt to use
        the native implementation.

        Args:
            image: 2D tensor of values between 0 and 1, either as np.ndarray or torch.Tensor.
            cmap: Colormap name.
            bytes: Whether to return the output as uint8 or floating point image.
            _force_method:
                Can be used to specify whether to use the native implementation (`"matplotlib"`), the efficient custom
                implementation of the select color maps (`"custom"`), or rely on autodetection (`None`, default).

        Returns:
            An RGB-colorized tensor corresponding to the input image.
        """
        if not (torch.is_tensor(image) or isinstance(image, np.ndarray)):
            raise ValueError("Argument must be a numpy array or torch tensor.")
        if _force_method not in (None, "matplotlib", "custom"):
            raise ValueError("_force_method must be either `None`, `'matplotlib'` or `'custom'`.")

        supported_cmaps = {
            "binary": [
                (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0),
            ],
            "Spectral": [  # Taken from matplotlib/_cm.py
                (0.61960784313725492, 0.003921568627450980, 0.25882352941176473),  # 0.0 -> [0]
                (0.83529411764705885, 0.24313725490196078, 0.30980392156862746),
                (0.95686274509803926, 0.42745098039215684, 0.2627450980392157),
                (0.99215686274509807, 0.68235294117647061, 0.38039215686274508),
                (0.99607843137254903, 0.8784313725490196, 0.54509803921568623),
                (1.0, 1.0, 0.74901960784313726),
                (0.90196078431372551, 0.96078431372549022, 0.59607843137254901),
                (0.6705882352941176, 0.8666666666666667, 0.64313725490196083),
                (0.4, 0.76078431372549016, 0.6470588235294118),
                (0.19607843137254902, 0.53333333333333333, 0.74117647058823533),
                (0.36862745098039218, 0.30980392156862746, 0.63529411764705879),  # 1.0 -> [K-1]
            ],
        }

        def method_matplotlib(image, cmap, bytes=False):
            import matplotlib

            arg_is_pt, device = torch.is_tensor(image), None
            if arg_is_pt:
                image, device = image.cpu().numpy(), image.device

            if cmap not in matplotlib.colormaps:
                raise ValueError(
                    f"Unexpected color map {cmap}; available options are: {', '.join(list(matplotlib.colormaps.keys()))}"
                )

            cmap = matplotlib.colormaps[cmap]
            out = cmap(image, bytes=bytes)  # [?,4]
            out = out[..., :3]  # [?,3]

            if arg_is_pt:
                out = torch.tensor(out, device=device)

            return out

        def method_custom(image, cmap, bytes=False):
            arg_is_np = isinstance(image, np.ndarray)
            if arg_is_np:
                image = torch.tensor(image)
            if image.dtype == torch.uint8:
                image = image.float() / 255
            else:
                image = image.float()

            is_cmap_reversed = cmap.endswith("_r")
            if is_cmap_reversed:
                cmap = cmap[:-2]

            if cmap not in supported_cmaps:
                raise ValueError(
                    f"Only {list(supported_cmaps.keys())} color maps are available without installing matplotlib."
                )

            cmap = supported_cmaps[cmap]
            if is_cmap_reversed:
                cmap = cmap[::-1]
            cmap = torch.tensor(cmap, dtype=torch.float, device=image.device)  # [K,3]
            K = cmap.shape[0]

            pos = image.clamp(min=0, max=1) * (K - 1)
            left = pos.long()
            right = (left + 1).clamp(max=K - 1)

            d = (pos - left.float()).unsqueeze(-1)
            left_colors = cmap[left]
            right_colors = cmap[right]

            out = (1 - d) * left_colors + d * right_colors

            if bytes:
                out = (out * 255).to(torch.uint8)

            if arg_is_np:
                out = out.numpy()

            return out

        if _force_method is None and torch.is_tensor(image) and cmap == "Spectral":
            return method_custom(image, cmap, bytes)

        out = None
        if _force_method != "custom":
            out = method_matplotlib(image, cmap, bytes)

        if _force_method == "matplotlib" and out is None:
            raise ImportError("Make sure to install matplotlib if you want to use a color map other than 'Spectral'.")

        if out is None:
            out = method_custom(image, cmap, bytes)

        return out

    @staticmethod
    def visualize_depth(
        depth: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.Tensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.Tensor],
        ],
        val_min: float = 0.0,
        val_max: float = 1.0,
        color_map: str = "Spectral",
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes depth maps, such as predictions of the `MarigoldDepthPipeline`.

        Args:
            depth (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray],
                List[torch.Tensor]]`): Depth maps.
            val_min (`float`, *optional*, defaults to `0.0`): Minimum value of the visualized depth range.
            val_max (`float`, *optional*, defaults to `1.0`): Maximum value of the visualized depth range.
            color_map (`str`, *optional*, defaults to `"Spectral"`): Color map used to convert a single-channel
                      depth prediction into colored representation.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with depth maps visualization.
        """
        if val_max <= val_min:
            raise ValueError(f"Invalid values range: [{val_min}, {val_max}].")

        def visualize_depth_one(img, idx=None):
            prefix = "Depth" + (f"[{idx}]" if idx else "")
            if isinstance(img, PIL.Image.Image):
                if img.mode != "I;16":
                    raise ValueError(f"{prefix}: invalid PIL mode={img.mode}.")
                img = np.array(img).astype(np.float32) / (2**16 - 1)
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim != 2:
                    raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if not torch.is_floating_point(img):
                    raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
            else:
                raise ValueError(f"{prefix}: unexpected type={type(img)}.")
            if val_min != 0.0 or val_max != 1.0:
                img = (img - val_min) / (val_max - val_min)
            img = MarigoldImageProcessor.colormap(img, cmap=color_map, bytes=True)  # [H,W,3]
            img = PIL.Image.fromarray(img.cpu().numpy())
            return img

        if depth is None or isinstance(depth, list) and any(o is None for o in depth):
            raise ValueError("Input depth is `None`")
        if isinstance(depth, (np.ndarray, torch.Tensor)):
            depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
            if isinstance(depth, np.ndarray):
                depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
            if not (depth.ndim == 4 and depth.shape[1] == 1):  # [N,1,H,W]
                raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
            return [visualize_depth_one(img[0], idx) for idx, img in enumerate(depth)]
        elif isinstance(depth, list):
            return [visualize_depth_one(img, idx) for idx, img in enumerate(depth)]
        else:
            raise ValueError(f"Unexpected input type: {type(depth)}")

    @staticmethod
    def export_depth_to_16bit_png(
        depth: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        val_min: float = 0.0,
        val_max: float = 1.0,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        def export_depth_to_16bit_png_one(img, idx=None):
            prefix = "Depth" + (f"[{idx}]" if idx else "")
            if not isinstance(img, np.ndarray) and not torch.is_tensor(img):
                raise ValueError(f"{prefix}: unexpected type={type(img)}.")
            if img.ndim != 2:
                raise ValueError(f"{prefix}: unexpected shape={img.shape}.")
            if torch.is_tensor(img):
                img = img.cpu().numpy()
            if not np.issubdtype(img.dtype, np.floating):
                raise ValueError(f"{prefix}: unexected dtype={img.dtype}.")
            if val_min != 0.0 or val_max != 1.0:
                img = (img - val_min) / (val_max - val_min)
            img = (img * (2**16 - 1)).astype(np.uint16)
            img = PIL.Image.fromarray(img, mode="I;16")
            return img

        if depth is None or isinstance(depth, list) and any(o is None for o in depth):
            raise ValueError("Input depth is `None`")
        if isinstance(depth, (np.ndarray, torch.Tensor)):
            depth = MarigoldImageProcessor.expand_tensor_or_array(depth)
            if isinstance(depth, np.ndarray):
                depth = MarigoldImageProcessor.numpy_to_pt(depth)  # [N,H,W,1] -> [N,1,H,W]
            if not (depth.ndim == 4 and depth.shape[1] == 1):
                raise ValueError(f"Unexpected input shape={depth.shape}, expecting [N,1,H,W].")
            return [export_depth_to_16bit_png_one(img[0], idx) for idx, img in enumerate(depth)]
        elif isinstance(depth, list):
            return [export_depth_to_16bit_png_one(img, idx) for idx, img in enumerate(depth)]
        else:
            raise ValueError(f"Unexpected input type: {type(depth)}")

    @staticmethod
    def visualize_normals(
        normals: Union[
            np.ndarray,
            torch.Tensor,
            List[np.ndarray],
            List[torch.Tensor],
        ],
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes surface normals, such as predictions of the `MarigoldNormalsPipeline`.

        Args:
            normals (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                Surface normals.
            flip_x (`bool`, *optional*, defaults to `False`): Flips the X axis of the normals frame of reference.
                      Default direction is right.
            flip_y (`bool`, *optional*, defaults to `False`):  Flips the Y axis of the normals frame of reference.
                      Default direction is top.
            flip_z (`bool`, *optional*, defaults to `False`): Flips the Z axis of the normals frame of reference.
                      Default direction is facing the observer.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with surface normals visualization.
        """
        flip_vec = None
        if any((flip_x, flip_y, flip_z)):
            flip_vec = torch.tensor(
                [
                    (-1) ** flip_x,
                    (-1) ** flip_y,
                    (-1) ** flip_z,
                ],
                dtype=torch.float32,
            )

        def visualize_normals_one(img, idx=None):
            img = img.permute(1, 2, 0)
            if flip_vec is not None:
                img *= flip_vec.to(img.device)
            img = (img + 1.0) * 0.5
            img = (img * 255).to(dtype=torch.uint8, device="cpu").numpy()
            img = PIL.Image.fromarray(img)
            return img

        if normals is None or isinstance(normals, list) and any(o is None for o in normals):
            raise ValueError("Input normals is `None`")
        if isinstance(normals, (np.ndarray, torch.Tensor)):
            normals = MarigoldImageProcessor.expand_tensor_or_array(normals)
            if isinstance(normals, np.ndarray):
                normals = MarigoldImageProcessor.numpy_to_pt(normals)  # [N,3,H,W]
            if not (normals.ndim == 4 and normals.shape[1] == 3):
                raise ValueError(f"Unexpected input shape={normals.shape}, expecting [N,3,H,W].")
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        elif isinstance(normals, list):
            return [visualize_normals_one(img, idx) for idx, img in enumerate(normals)]
        else:
            raise ValueError(f"Unexpected input type: {type(normals)}")

    @staticmethod
    def visualize_uncertainty(
        uncertainty: Union[
            np.ndarray,
            torch.Tensor,
            List[np.ndarray],
            List[torch.Tensor],
        ],
        saturation_percentile=95,
    ) -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
        """
        Visualizes dense uncertainties, such as produced by `MarigoldDepthPipeline` or `MarigoldNormalsPipeline`.

        Args:
            uncertainty (`Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]`):
                Uncertainty maps.
            saturation_percentile (`int`, *optional*, defaults to `95`):
                Specifies the percentile uncertainty value visualized with maximum intensity.

        Returns: `PIL.Image.Image` or `List[PIL.Image.Image]` with uncertainty visualization.
        """

        def visualize_uncertainty_one(img, idx=None):
            prefix = "Uncertainty" + (f"[{idx}]" if idx else "")
            if img.min() < 0:
                raise ValueError(f"{prefix}: unexected data range, min={img.min()}.")
            img = img.squeeze(0).cpu().numpy()
            saturation_value = np.percentile(img, saturation_percentile)
            img = np.clip(img * 255 / saturation_value, 0, 255)
            img = img.astype(np.uint8)
            img = PIL.Image.fromarray(img)
            return img

        if uncertainty is None or isinstance(uncertainty, list) and any(o is None for o in uncertainty):
            raise ValueError("Input uncertainty is `None`")
        if isinstance(uncertainty, (np.ndarray, torch.Tensor)):
            uncertainty = MarigoldImageProcessor.expand_tensor_or_array(uncertainty)
            if isinstance(uncertainty, np.ndarray):
                uncertainty = MarigoldImageProcessor.numpy_to_pt(uncertainty)  # [N,1,H,W]
            if not (uncertainty.ndim == 4 and uncertainty.shape[1] == 1):
                raise ValueError(f"Unexpected input shape={uncertainty.shape}, expecting [N,1,H,W].")
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        elif isinstance(uncertainty, list):
            return [visualize_uncertainty_one(img, idx) for idx, img in enumerate(uncertainty)]
        else:
            raise ValueError(f"Unexpected input type: {type(uncertainty)}")


class AbsoluteDepthScaler:
    @staticmethod
    def masked_min_max(
        input_tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        计算掩码为True的元素, 对应的最大值和最小值。

        参数:
        input_tensor (torch.Tensor): 形状为 [B, ...] 的输入张量。
        mask (torch.Tensor): 形状为 [B, ...] 的布尔掩码张量。

        返回:
        min_values (torch.Tensor): 形状为 [B,] 的最小值张量。
        max_values (torch.Tensor): 形状为 [B,] 的最大值张量。
        """
        # 将掩码应用到输入张量上，只保留掩码为True的元素
        if mask is not None:
            masked_tensor = input_tensor.masked_fill(~mask, float('-inf'))
            masked_tensor_min = input_tensor.masked_fill(~mask, float('inf'))
        else:
            masked_tensor = input_tensor
            masked_tensor_min = input_tensor
        
        # 计算沿着非B轴的最大值和最小值
        min_values, _ = torch.min(masked_tensor_min.view(masked_tensor_min.size(0), -1), dim=1)
        max_values, _ = torch.max(masked_tensor.view(masked_tensor.size(0), -1), dim=1)

        return min_values, max_values

    @staticmethod
    def normalize(
        depth: torch.Tensor,
        scale: float,
        min_value: Optional[torch.Tensor] = None,
        max_value: Optional[torch.Tensor] = None,
        mask_for_minmax: Optional[torch.Tensor] = None,
        eps: float = 1e-9,
    ):
        """
        Args:
            depth: [B, ...], torch.FloatTensor, [0.0, +inf]
            scale: float
            min_value: [B], torch.FloatTensor, [0.0, +inf]
            max_value: [B], torch.FloatTensor, [0.0, +inf]
            mask_for_minmax: [B, ...], torch.BoolTensor, [0, 1], True: available, False: unavailable
        
        Returns:
            normalized_depth: [B, ...], torch.FloatTensor, [-1.0, 1.0]
            min_value: [B,], torch.FloatTensor, [0.0, +inf]
            max_value: [B,], torch.FloatTensor, [0.0, +inf]
        """
        if min_value is None or max_value is None:
            min_value, max_value = AbsoluteDepthScaler.masked_min_max(depth, mask_for_minmax)
        
        viewed_min_value = min_value.view(-1, *[1] * (depth.dim() - 1))
        viewed_max_value = max_value.view(-1, *[1] * (depth.dim() - 1))
        normalized_depth = ((depth - viewed_min_value) / (viewed_max_value - viewed_min_value + eps) - 0.5) * scale * 2

        return normalized_depth, min_value, max_value
    
    @staticmethod
    def denormalize(
        normalized_depth: torch.Tensor,
        scale: float,
        min_value: torch.Tensor,
        max_value: torch.Tensor,
    ):
        """
        Args:
            normalized_depth: [B, ...], torch.FloatTensor, [-1.0, 1.0]
            min_value: [B,], torch.FloatTensor, [0.0, +inf]
            max_value: [B,], torch.FloatTensor, [0.0, +inf]
            scale: float
        
        Returns:
            depth: [B, ...], torch.FloatTensor, [0.0, +inf]
        """
        assert min_value.shape[0] == max_value.shape[0] == normalized_depth.shape[0], 'The batch size should be the same.'
        viewed_min_value = min_value.view(-1, *[1] * (normalized_depth.dim() - 1))
        viewed_max_value = max_value.view(-1, *[1] * (normalized_depth.dim() - 1))
        
        depth = (normalized_depth + scale) / (scale * 2)  # [-1.0, 1.0] -> [s-1.0, s+1.0] -> []
        depth = depth * (viewed_max_value - viewed_min_value) + viewed_min_value  # [-s, s], s<=1.0 -> [0.0, +inf]

        return depth


def depth_to_z_distance(depth_map: np.ndarray | torch.Tensor, fov_x: float, fov_y: float) -> np.ndarray:
    """
    将深度图从直线距离转换为z距离。

    :param depth_map: 输入深度图，形状为 (..., H, W, 1) 的 np.ndarray
    :param fov_x: 相机的水平视场角（单位：度）。
    :param fov_y: 相机的垂直视场角（单位：度）。
    :return: 转换后的 z 距离图，形状为 (..., H, W, 1)。
    """
    # 图像尺寸
    *_, H, W, _ = depth_map.shape
    
    # 转换 FOV 为弧度
    fov_x_rad = np.radians(fov_x)
    fov_y_rad = np.radians(fov_y)
    
    # 计算焦距（假设相机为透视投影模型）
    f_x = W / (2 * np.tan(fov_x_rad / 2))  # 水平焦距
    f_y = H / (2 * np.tan(fov_y_rad / 2))  # 垂直焦距
    
    # 像素网格的坐标 [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 像素坐标归一化，中心化到 [-1, 1] 范围
    u_normalized = (u - W / 2) / f_x
    v_normalized = (v - H / 2) / f_y
    
    # 计算与主光轴的夹角（方向向量和 Z 轴的夹角）
    angles = np.arctan(np.sqrt(u_normalized**2 + v_normalized**2))

    # 扩展批次维度
    angles = repeat(angles, 'h w -> 1 h w 1')
    # angles = repeat(angles, 'h w -> b h w 1', b=B)
    
    # 使用深度图和角度计算z值
    if isinstance(depth_map, torch.Tensor):
        z_distance = depth_map * torch.from_numpy(np.cos(angles)).to(depth_map.device)
    else:
        z_distance = depth_map * np.cos(angles)
    
    # 返回形状为 (B, H, W, 1) 的z距离图
    return z_distance


def z_distance_to_depth(z_distance: np.ndarray, fov_x: float, fov_y: float) -> np.ndarray:
    """
    将z距离图转换为直线距离。

    :param z_distance: 输入 z 距离图，形状为 (..., H, W, 1) 的 np.ndarray
    :param fov_x: 相机的水平视场角（单位：度）。
    :param fov_y: 相机的垂直视场角（单位：度）。
    :return: 转换后的深度图，形状为 (B, H, W, 1)。
    """
    # 图像尺寸
    *_, H, W, C = z_distance.shape
    # assert C == 1, "输入的 z 距离图必须是单通道的"
    
    # 转换 FOV 为弧度
    fov_x_rad = np.radians(fov_x)
    fov_y_rad = np.radians(fov_y)
    
    # 计算焦距（假设相机为透视投影模型）
    f_x = W / (2 * np.tan(fov_x_rad / 2))  # 水平焦距
    f_y = H / (2 * np.tan(fov_y_rad / 2))  # 垂直焦距
    
    # 像素网格的坐标 [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 像素坐标归一化，中心化到 [-1, 1] 范围
    u_normalized = (u - W / 2) / f_x
    v_normalized = (v - H / 2) / f_y
    
    # 计算与主光轴的夹角（方向向量和 Z 轴的夹角）
    angles = np.arctan(np.sqrt(u_normalized**2 + v_normalized**2))
    
    # 扩展批次维度
    angles = repeat(angles, 'h w -> 1 h w 1')
    angles = np.broadcast_to(angles, z_distance.shape)
    
    # 使用z距离图和角度计算深度值
    depth_map = z_distance / np.cos(angles)
    
    # 返回形状为 (B, H, W, 1) 的深度图
    return depth_map


@torch.no_grad()
def eval_depth(pred, target, eps=1e-9):
    # Borrowed from https://huggingface.co/blog/Isayoften/monocular-depth-estimation-guide
    assert pred.shape == target.shape

    pred = pred.flatten()
    target = target.flatten()

    thresh = torch.max(((target + 1.0) / (pred + 1.0)), ((pred + 1.0) / (target + 1.0)))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred - target

    abs_rel = torch.mean(torch.abs(diff) / (target + 1.0))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    mae = torch.mean(torch.abs(diff))

    # diff_log = torch.log(pred) - torch.log(target)
    # silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    # return {'d1': d1.detach(), 'abs_rel': abs_rel.detach(),'rmse': rmse.detach(), 'mae': mae.detach(), 'silog':silog.detach()}
    return {
        'd1': d1.detach().item(),
        'abs_rel': abs_rel.detach().item(),
        'rmse': rmse.detach().item(),
        'mae': mae.detach().item(),
    }
