#include "pibr.h"

// parallel ibr intervals extraction
__global__ void pibr_extraction(unsigned char *a, int height, int width, int *interval_row, introw_t *ir)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int temp;

    if (thread_id < height) // each thread represents an image row
    {
        int interval_found = 0;
        interval_row[thread_id] = 0;

        for (int w = 0; w < width - 1; w++)
        {
            if (a[thread_id * width + w] && !interval_found)
            {
                temp = interval_row[thread_id];
                interval_found = 1;

                ir[thread_id * height + temp].x1 = w;
                interval_row[thread_id] = temp;
            }
            else if (!a[thread_id * width + w] && interval_found)
            {
                interval_found = 0;
                temp = interval_row[thread_id];
                ir[thread_id * height + temp].x2 = w - 1;
                temp++;
                interval_row[thread_id] = temp;
            }
        }

        if (!a[thread_id * width + width - 1])
        {
            if (interval_found)
            {
                temp = interval_row[thread_id];
                ir[thread_id * height + temp].x2 = width - 2;
                temp++;
                interval_row[thread_id] = temp;
            }
        }
        else
        {
            temp = interval_row[thread_id];
            if (!interval_found)
            {
                temp = interval_row[thread_id];
                ir[thread_id * height + temp].x1 = width - 1;
            }
            ir[thread_id * height + temp].x2 = width - 1;
            temp++;
            interval_row[thread_id] = temp;
        }
    }
}

// parallel ibr intervals matching, creating blocks
__global__ void pibr_block_creation(introw_t *ir, int height, int width, blocktype *block, int *interval_row, int *is_block, int *blockno)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    int j_prev = 0, j_curr = 0;

    if (thread_id < height) // each thread represents an image row
    {
        blockno[thread_id] = 0;
        if (interval_row[(thread_id - 1)] > 0) // if previous row has interval
        {
            for (int i = 0; i < interval_row[thread_id]; i++) // for every interval in current row
            {
                is_block[(thread_id)*height + i] = 1;
                for (int j = j_curr; j < interval_row[(thread_id)-1]; j++) // for every interval in previous row (interval_row = number of intervals)
                {
                    j_prev = j;
                    if (ir[(thread_id)*height + i].x1 < ir[(thread_id - 1) * height + j].x1)
                    {
                        break;
                    }
                    else if (ir[(thread_id)*height + i].x1 == ir[(thread_id - 1) * height + j].x1 && ir[(thread_id)*height + i].x2 == ir[(thread_id - 1) * height + j].x2)
                    {
                        is_block[(thread_id)*height + i] = 0;
                        break;
                    }
                }

                if (is_block[(thread_id)*height + i]) // create block
                {
                    block[(thread_id)*height + blockno[thread_id]].x1 = ir[(thread_id)*height + i].x1;
                    block[(thread_id)*height + blockno[thread_id]].x2 = ir[(thread_id)*height + i].x2;
                    block[(thread_id)*height + blockno[thread_id]].y1 = thread_id;
                    block[(thread_id)*height + blockno[thread_id]].y2 = thread_id;
                    blockno[thread_id]++;
                }

                j_curr = j_prev;
            }
        }
        else // if previous row does not hav interval (e.g. first line of image)
        {
            for (int i = 0; i < interval_row[thread_id]; i++)
            {
                is_block[(thread_id)*height + i] = 1;
                block[(thread_id)*height + blockno[thread_id]].x1 = ir[(thread_id)*height + i].x1;
                block[(thread_id)*height + blockno[thread_id]].x2 = ir[(thread_id)*height + i].x2;
                block[(thread_id)*height + blockno[thread_id]].y1 = thread_id;
                // block[(thread_id)*height + blockno[thread_id]].y2 = thread_id;
                blockno[thread_id]++;
            }
        }
    }
}

// find y2 of created blocks
__global__ void pibr_block_creation2(introw_t *ir, int height, int width, blocktype *block, int *interval_row, int *is_block, int *blockno)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    int temp;
    if (thread_id < height) // each thread represents an image row
    {
        for (int i = 0; i < blockno[thread_id]; i++)
        {
            temp = thread_id;
            for (int k = thread_id + 1; k - temp == 1 && k < height; k++)
            {
                for (int jjj = 0; jjj < interval_row[(k)]; jjj++) // for every interval in next row
                {
                    if (block[(thread_id)*height + i].x1 == ir[(k)*height + jjj].x1 && block[(thread_id)*height + i].x2 == ir[(k)*height + jjj].x2)
                    {
                        temp = k;
                        break;
                    }
                    else if (block[(thread_id)*height + i].x1 < ir[(k)*height + jjj].x1)
                    {

                        break;
                    }
                }
            }
            block[(thread_id)*height + i].y2 = temp;
        }
    }
}
