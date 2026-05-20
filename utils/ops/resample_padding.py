import os
import torch
import numpy as np

from . import custom_ops
from . import misc
from . import grid_sample_gradfix


#----------------------------------------------------------------------------
def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

#----------------------------------------------------------------------------

_plugin = None

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='resample_padding_plugin',
            sources=['resample_padding.cpp', 'resample_padding.cu'],
            headers=['resample_padding.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
    return True

#----------------------------------------------------------------------------

def resample_padding(x, sampling_rate, backwards=False, impl='cuda'):
    r"""Resample padding function.

    Replaces padded regions of cube faces with content from neighboring faces.

    Args:
        x:              Input activation tensor.
        sampling_rate:  Side-length of visible cube face, in pixels.
                        Padded regions are determined by image size - sampling_rate.
        backwards:      Whether to perform the forward pass or the backward pass.

    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _resample_padding_cuda(sampling_rate, backwards=backwards).apply(x)
    return _resample_padding_ref(x, sampling_rate, backwards=backwards)

#----------------------------------------------------------------------------

@misc.profiled_function
def _resample_padding_ref(x, sampling_rate, backwards=False):
    assert isinstance(x, torch.Tensor) and x.ndim == 4

    height, width = x.shape[2:]
    assert height >= sampling_rate and width >= sampling_rate
    # Do nothing if there's no padding
    if height == int(sampling_rate) and width == int(sampling_rate):
        return x
    # Don't support backward pass for ref impl
    assert backwards == False

    x = x.view([-1, 6, *x.shape[1:]])

    total_padx = width - int(sampling_rate)
    total_pady = height - int(sampling_rate)
    padx0 = total_padx // 2
    padx1 = total_padx - padx0
    pady0 = total_pady // 2
    pady1 = total_pady - pady0

    # Construct cube matrices
    cube_mtx = torch.eye(3,3,device=x.device).unsqueeze(0).tile(6,1,1)
    cube_mtx[0] = matrix(
        [ 1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0,-1])
    cube_mtx[1] = matrix(
        [ 0, 0, 1],
        [ 0,-1, 0],
        [ 1, 0, 0])
    cube_mtx[2] = matrix(
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1])
    cube_mtx[3] = matrix(
        [ 0, 0,-1],
        [ 0,-1, 0],
        [-1, 0, 0])
    cube_mtx[4] = matrix(
        [ 1, 0, 0],
        [ 0, 0, 1],
        [ 0,-1, 0])
    cube_mtx[5] = matrix(
        [ 1, 0, 0],
        [ 0, 0,-1],
        [ 0, 1, 0])
    cube_mtx_inv = cube_mtx.permute(0,2,1)

    # Construct padding mask
    mask = torch.zeros_like(x[0, 0, 0], device=x.device).to(torch.bool)
    mask[:,:padx0] = True
    mask[:,-padx1:] = True
    mask[:pady0,:] = True
    mask[-pady1:,:] = True

    # Construct sampling grid.
    theta = torch.eye(2, 3, device=x.device)
    theta[0][0] = width / sampling_rate
    theta[1][1] = height / sampling_rate
    theta[0][2] = (padx1 - padx0) / sampling_rate
    theta[1][2] = (pady1 - pady0) / sampling_rate
    grid = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, height, width], align_corners=False)
    grid = grid[:,mask]  # [1,P,xy]
    grid = torch.cat([grid, torch.ones_like(grid[:,:,:1])], 2)  # [1,P,xyz]
    grid = (cube_mtx.unsqueeze(1) @ grid.unsqueeze(3)).squeeze(3)  # [6,P,xyz]
    grid = grid.view([-1,3])  # [6*P,xyz]

    # Determine each sample's cube face
    Agrid = torch.abs(grid)
    fb_idx = (Agrid[:,2] >= Agrid[:,0]) & (Agrid[:,2] >= Agrid[:,1])
    cube_idx = ((fb_idx & (grid[:,2] < 0)) * 0) | ((fb_idx & (grid[:,2] > 0)) * 2)
    tb_idx = (~fb_idx) & (Agrid[:,1] >= Agrid[:,0])
    cube_idx |= ((tb_idx & (grid[:,1] > 0)) * 4) | ((tb_idx & (grid[:,1] < 0)) * 5)
    lr_idx = ~(fb_idx | tb_idx)
    cube_idx |= ((lr_idx & (grid[:,0] > 0)) * 1) | ((lr_idx & (grid[:,0] < 0)) * 3)

    # Invert cube transform, reproject to new face
    grid = (cube_mtx_inv[cube_idx] @ grid.unsqueeze(2)).squeeze(2)  # [6*P,xyz]
    grid /= grid[:,2:]
    # Replace z with "depth" of cube face in image stack
    grid[:,2] = cube_idx * 2 / 6 - 5 / 6

    # Rescale to padded image dimensions
    grid[:,0] = (grid[:,0] - (padx1 - padx0) / sampling_rate) * sampling_rate / width
    grid[:,1] = (grid[:,1] - (pady1 - pady0) / sampling_rate) * sampling_rate / width

    # Reshape for grid sampling
    grid = grid.reshape([1,6,-1,1,3]).tile([x.shape[0],1,1,1,1]).to(x.dtype)  # [1,6,P,1,xyz]
    x = x.permute(0,2,1,3,4)  # [B,C,6,H,W]

    # Resample
    x = x.masked_scatter(mask, grid_sample_gradfix.grid_sample(x, grid).squeeze(4))
    x = x.permute(0,2,1,3,4)  # [B,6,C,H,W]

    # x = x.view([-1, *x.shape[2:]])
    x = x.reshape([-1, *x.shape[2:]])
    return x


#----------------------------------------------------------------------------

_resample_padding_cuda_cache = dict()

def _resample_padding_cuda(sampling_rate, backwards=False):
    """Fast CUDA implementation of 'resample_padding()' using custom ops.
    """
    # Parse arguments.
    assert isinstance(sampling_rate, int) and sampling_rate >= 1

    # Lookup from cache.
    key = (sampling_rate, backwards)
    if key in _resample_padding_cuda_cache:
        return _resample_padding_cuda_cache[key]

    # Forward op.
    class ResamplePadding(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            x = x.contiguous()
            y = _plugin.resample_padding(x, sampling_rate, backwards)
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            dx = None
            if ctx.needs_input_grad[0]:
                dx = _resample_padding_cuda(sampling_rate=sampling_rate, backwards=(not backwards)).apply(dy)
            return dx

    # Add to cache.
    _resample_padding_cuda_cache[key] = ResamplePadding
    return ResamplePadding

#----------------------------------------------------------------------------
