#include "pibr.h"

__global__ void cheby_poly(float *poly, int *dim_order)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < dim_order[0]) // each thread represents an image row
    {
        poly[thread_id] = 1;
        poly[dim_order[0] + thread_id] = (2.0 * thread_id + 1 - dim_order[0]) / (float)dim_order[0];
        for (int n = 2; n < dim_order[1]; n++)
        {
            poly[n * dim_order[0] + thread_id] = ((2 * n - 1) * poly[dim_order[0] + thread_id] * poly[(n - 1) * dim_order[0] + thread_id] - (n - 1) * (1 - (n - 1) * (n - 1) / (dim_order[0] * dim_order[0])) * poly[(n - 2) * dim_order[0] + thread_id]) / n;
        }
    }
}

__global__ void tt_calculation(int *dim_order, float *ttx)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id < dim_order[0]) // each thread represents an image row
    {
        // in 2D ttx[n][x], n = order moment, x = x-axis values
        ttx[thread_id] = 1;
        ttx[dim_order[0] + thread_id] = (2.0 * thread_id + 1 - dim_order[0]) / (float)dim_order[0];
        for (int n = 2; n < dim_order[1]; n++)
        {
            ttx[n * dim_order[0] + thread_id] = ((2 * n - 1) * ttx[dim_order[0] + thread_id] * ttx[(n - 1) * dim_order[0] + thread_id] - (n - 1) * (1 - (n - 1) * (n - 1) / (dim_order[0] * dim_order[0])) * ttx[(n - 2) * dim_order[0] + thread_id]) / n;
            // printf("ttx[%d] = %f \n", n*dim_order[1] + x, ttx[n*dim_order[1] + x]);
        }
    }
}

__global__ void tb_calculation(int *dim_order, blocktype *block, float *ttx, float *stx, float *sty, int blockno, float *tb, float *r)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (thread_id == 0) // syntelestis kanonikopoiisis r
    {
        r[0] = dim_order[0];
        for (int n = 1; n < dim_order[1]; n++)
        {
            r[n] = r[n - 1] * (1 - n * n / (dim_order[0] * dim_order[0]));
        }
        for (int n = 0; n < dim_order[1]; n++)
        {
            r[n] /= (2 * n + 1);
            // printf("r[%d] = %f\n", n, r[n]);
        }
    }

    if (thread_id < blockno)
    {

        // in 2D stx[p][k], sty[q][k] p,q = order moment, k = blockno
        for (int p = 0; p < dim_order[1]; p++)
        {
            stx[p * blockno + thread_id] = 0;
            for (int x = block[thread_id].x1; x <= block[thread_id].x2; x++)
            {
                stx[p * blockno + thread_id] += ttx[p * dim_order[0] + x];
            }
        }

        for (int q = 0; q < dim_order[1]; q++)
        {
            sty[q * blockno + thread_id] = 0;
            for (int y = block[thread_id].y1; y <= block[thread_id].y2; y++)
            {
                sty[q * blockno + thread_id] += ttx[q * dim_order[0] + y];
            }
        }

        for (int p = 0; p < dim_order[1]; p++)
        {
            for (int q = 0; q < dim_order[1]; q++)
            {
                atomicAdd(&tb[p * dim_order[1] + q], stx[p * blockno + thread_id] * sty[q * blockno + thread_id] / (r[p] * r[q]));
                // tb[p*dim_order[1] + q] = stx[p*blockno + thread_id] * sty[q*blockno + thread_id] / (r[p] * r[q]);
            }
        }
    }
}
