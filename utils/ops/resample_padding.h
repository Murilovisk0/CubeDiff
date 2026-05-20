#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct resample_padding_kernel_params
{
    const void* x;        // Input
    void*       y;        // Output

    int         sampling_rate;
    bool        backwards;

    int         sizeX;    // Width
    int         sizeY;    // Height
    int         sizeC;    // Channels
    int         sizeF;    // Faces == 6
    int         sizeB;    // Batch size

    int         strideX;  // Strides
    int         strideY;
    int         strideC;
    int         strideF;
    int         strideB;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_resample_padding_kernel(const resample_padding_kernel_params& p);

//------------------------------------------------------------------------
