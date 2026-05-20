#include <c10/util/Half.h>
#include <THC/THCAtomics.cuh>
#include "resample_padding.h"

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

//------------------------------------------------------------------------
// CUDA kernel.

template <class T, bool B>
__global__ void resample_padding_kernel(resample_padding_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;

    // Get padded region
    int total_padx = p.sizeX - p.sampling_rate;
    int total_pady = p.sizeY - p.sampling_rate;
    int padx0 = total_padx / 2;
    int padx1 = total_padx - padx0;
    int pady0 = total_pady / 2;
    int pady1 = total_pady - pady0;

    // Get pixel coordinates from thread & block ID
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z % p.sizeC;
    int f = (blockIdx.z / p.sizeC) % p.sizeF;
    int b = (blockIdx.z / (p.sizeC * p.sizeF)) % p.sizeB;

    // If in padded region
    if (x < padx0 || x >= padx0 + p.sampling_rate || y < pady0 || y >= pady0 + p.sampling_rate) {
        // Get sampling vector in normalized space
        scalar_t vx = ((x + (scalar_t)0.5) * 2 - p.sizeX + padx1 - padx0) / p.sampling_rate;
        scalar_t vy = ((y + (scalar_t)0.5) * 2 - p.sizeY + pady1 - pady0) / p.sampling_rate;
        scalar_t vz = (scalar_t)1.0;

        // Rotate to correct face
        if (f == 0) {
            vy = -vy;
            vz = -vz;
        } else if (f == 1) {
            scalar_t tmp = vx;
            vx = vz;
            vy = -vy;
            vz = tmp;
        } else if (f == 2) {
            vx = -vx;
            vy = -vy;
        } else if (f == 3) {
            scalar_t tmp = vx;
            vx = -vz;
            vy = -vy;
            vz = -tmp;
        } else if (f == 4) {
            scalar_t tmp = vy;
            vy = vz;
            vz = -tmp;
        } else {  // f == 5
            scalar_t tmp = vy;
            vy = -vz;
            vz = tmp;
        }

        // Get new face index
        int f_s;
        if (abs(vz) >= abs(vx) && abs(vz) >= abs(vy))
            if (vz < 0) f_s = 0; else f_s = 2;
        else if (abs(vy) >= abs(vx))
            if (vy > 0) f_s = 4; else f_s = 5;
        else
            if (vx > 0) f_s = 1; else f_s = 3;

        // Un-rotate by new face
        if (f_s == 0) {
            vy = -vy;
            vz = -vz;
        } else if (f_s == 1) {
            scalar_t tmp = vx;
            vx = vz;
            vy = -vy;
            vz = tmp;
        } else if (f_s == 2) {
            vx = -vx;
            vy = -vy;
        } else if (f_s == 3) {
            scalar_t tmp = vx;
            vx = -vz;
            vy = -vy;
            vz = -tmp;
        } else if (f_s == 4) {
            scalar_t tmp = vy;
            vy = -vz;
            vz = tmp;
        } else {  // f_s == 5
            scalar_t tmp = vy;
            vy = vz;
            vz = -tmp;
        }

        // Reproject onto cube
        vx /= vz;
        vy /= vz;

        // Rescale to pixel space
        vx = (vx * p.sampling_rate + p.sizeX + padx0 - padx1) / 2 - (scalar_t)0.5;
        vy = (vy * p.sampling_rate + p.sizeY + pady0 - pady1) / 2 - (scalar_t)0.5;

        // Get sampling coordinates
        int ix_nw = (int)floor(vx);
        int iy_nw = (int)floor(vy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;
        scalar_t nw = (ix_se - vx) * (iy_se - vy);
        scalar_t ne = (vx - ix_sw) * (iy_sw - vy);
        scalar_t sw = (ix_ne - vx) * (vy - iy_ne);
        scalar_t se = (vx - ix_nw) * (vy - iy_nw);

        if (!B) {
            // Gather
            // output[b,f,c,y,x] += input[b,f_s,c,iy_nw,ix_nw] * nw
            // output[b,f,c,y,x] += input[b,f_s,c,iy_ne,ix_ne] * ne
            // output[b,f,c,y,x] += input[b,f_s,c,iy_sw,ix_sw] * sw
            // output[b,f,c,y,x] += input[b,f_s,c,iy_se,ix_se] * se
            if (x < p.sizeX && y < p.sizeY) {
                *((T*)p.y + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) +=
                    *((T*)p.x + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_nw*p.strideY + ix_nw*p.strideX) * nw;
                *((T*)p.y + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) +=
                    *((T*)p.x + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_ne*p.strideY + ix_ne*p.strideX) * ne;
                *((T*)p.y + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) +=
                    *((T*)p.x + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_sw*p.strideY + ix_sw*p.strideX) * sw;
                *((T*)p.y + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) +=
                    *((T*)p.x + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_se*p.strideY + ix_se*p.strideX) * se;
            }
        } else {
            // Scatter
            // output[b,f_s,c,iy_nw,ix_nw] += input[b,f,c,y,x] * nw
            // output[b,f_s,c,iy_ne,ix_ne] += input[b,f,c,y,x] * ne
            // output[b,f_s,c,iy_sw,ix_sw] += input[b,f,c,y,x] * sw
            // output[b,f_s,c,iy_se,ix_se] += input[b,f,c,y,x] * se
            if (x < p.sizeX && y < p.sizeY) {
                gpuAtomicAdd(
                    ((T*)p.y + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_nw*p.strideY + ix_nw*p.strideX),
                    *((T*)p.x + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) * nw);
                gpuAtomicAdd(
                    ((T*)p.y + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_ne*p.strideY + ix_ne*p.strideX),
                    *((T*)p.x + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) * ne);
                gpuAtomicAdd(
                    ((T*)p.y + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_sw*p.strideY + ix_sw*p.strideX),
                    *((T*)p.x + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) * sw);
                gpuAtomicAdd(
                    ((T*)p.y + b*p.strideB + f_s*p.strideF + c*p.strideC + iy_se*p.strideY + ix_se*p.strideX),
                    *((T*)p.x + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX) * se);
            }
        }

    // Not in padded region
    } else {
        // output[b,f,c,y,x] += input[b,f,c,y,x]
        if (x < p.sizeX && y < p.sizeY) {
            gpuAtomicAdd(
                ((T*)p.y + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX),
                *((T*)p.x + b*p.strideB + f*p.strideF + c*p.strideC + y*p.strideY + x*p.strideX));
        }
    }
}

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_resample_padding_kernel(const resample_padding_kernel_params& p)
{
    if (!p.backwards) return (void*)resample_padding_kernel<T, false>;
    if (p.backwards)  return (void*)resample_padding_kernel<T, true>;
    return NULL;
}

//------------------------------------------------------------------------
// Template specializations.

template void* choose_resample_padding_kernel<double>     (const resample_padding_kernel_params& p);
template void* choose_resample_padding_kernel<float>      (const resample_padding_kernel_params& p);
template void* choose_resample_padding_kernel<c10::Half>  (const resample_padding_kernel_params& p);

//------------------------------------------------------------------------
