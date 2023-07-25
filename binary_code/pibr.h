#ifndef PIBR_H
#define PIBR_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#define BLOCK_SIZE 32
#define GRID_SIZE (width / BLOCK_SIZE + (height % BLOCK_SIZE == 0 ? 0 : 1))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define ABS(a) (((a) < 0) ? -(a) : (a))
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s in file %s at line %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

typedef struct
{
    int b, x1, x2;
} introw_t;

typedef struct
{
    int x1, y1, x2, y2;
} blocktype;

void read(char *file, unsigned char *a, int image_height, int image_width);

void write(int height, int *is_block, blocktype *block, int *host_interval_row, int *host_check);

void write_poly(float *poly, int *dim_order);

// void write_tt(float *tt, int *dim_order, blocktype *block, int *is_block);

// void write_st(float *st, int *dim_order, blocktype *block, int *is_block);

// void write_tb(float *tb, float *stx, float *sty, int *dim_order, blocktype *block, int *is_block);

void write1d(float *output_array, int image_height, int image_width);

void write2d(unsigned char **output_array, int image_height, int image_width);

void make_block_image(int width, int height, unsigned char **blockimg, blocktype *block, int *is_block, int *interval_row);

__global__ void calculate_tb_and_r(float *r, int *dim_order, float *stx, float *sty, float *tb, int blockno);

void cheby_2d_mom(float *r, int *dim_order, float *tt, float *t2d, float *tb, unsigned char *img);

void cheby_rebuild(int height, int width, int *dim_order, float *reb_img, float *tt, float *tb);

void convert_block_indexing(blocktype *block, int *is_block, int height, int width, blocktype *a, int *interval_row, int total, int *blockno);

double nire_calculation(unsigned char *original_image, float *reconstructed_image, int height, int width);

__device__ static float atomicMax(float *address, float val);

__global__ void pibr_extraction(unsigned char *a, int height, int width, int *interval_row, introw_t *ir);

__global__ void pibr_block_creation(introw_t *ir, int height, int width, blocktype *block, int *interval_row, int *is_block, int *blockno);

__global__ void pibr_block_creation2(introw_t *ir, int height, int width, blocktype *block, int *interval_row, int *is_block, int *blockno);

__global__ void cheby_poly(float *poly, int *dim_order);

__global__ void tt_calculation(int *dim_order, blocktype *block, float *ttx, float *stx, float *sty);

__global__ void tb_calculation(int *dim_order, blocktype *block, float *ttx, float *stx, float *sty, int blockno, float *tb, float *r);
__global__ void moments_calculation(int *dim_order, blocktype *block, float *ttx, float *stx, float *sty, int blockno, float *tb, float *r);

#endif