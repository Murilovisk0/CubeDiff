#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "resample_padding.h"

//------------------------------------------------------------------------

static bool has_same_layout(torch::Tensor x, torch::Tensor y)
{
    if (x.dim() != y.dim())
        return false;
    for (int64_t i = 0; i < x.dim(); i++)
    {
        if (x.size(i) != y.size(i))
            return false;
        if (x.size(i) >= 2 && x.stride(i) != y.stride(i))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor resample_padding(torch::Tensor x, int sampling_rate, bool backwards)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(sampling_rate >= 0, "sampling rate must be non-negative");
    // Validate layout.
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    torch::Tensor y = torch::zeros_like(x);
    TORCH_CHECK(has_same_layout(y, x), "y must have the same layout as x");

    // Initialize CUDA kernel parameters.
    resample_padding_kernel_params p;
    p.x             = x.data_ptr();
    p.y             = y.data_ptr();
    p.sampling_rate = sampling_rate;
    p.backwards     = backwards;
    p.sizeX = (int)x.size(3);
    p.sizeY = (int)x.size(2);
    p.sizeC = (int)x.size(1);
    p.sizeF = 6;
    p.sizeB = (int)x.size(0) / 6;
    p.strideX = (int)x.stride(3);
    p.strideY = (int)x.stride(2);
    p.strideC = (int)x.stride(1);
    p.strideF = (int)x.stride(0);
    p.strideB = (int)x.stride(0) * 6;

    // Choose CUDA kernel.
    void* kernel;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "resample_padding_cuda", [&]
    {
        kernel = choose_resample_padding_kernel<scalar_t>(p);
    });
    TORCH_CHECK(kernel, "no CUDA kernel found for the specified activation func");

    // Launch CUDA kernel.
    int maxBlockSize = 32;
    dim3 gridSize = dim3((p.sizeX - 1) / maxBlockSize + 1,
                              (p.sizeY - 1) / maxBlockSize + 1,
                              p.sizeC * p.sizeF * p.sizeB);
    dim3 blockSize = dim3(((p.sizeX - 1) / gridSize.x + 1 + 3) & ~3,
                               ((p.sizeY - 1) / gridSize.y + 1 + 3) & ~3,
                               1);
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("resample_padding", &resample_padding);
}

//------------------------------------------------------------------------
