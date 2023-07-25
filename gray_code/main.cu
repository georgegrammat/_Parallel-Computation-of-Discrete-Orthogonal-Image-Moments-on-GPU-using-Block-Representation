#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include "pibr.h"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Give arguments: image, height, width, max_order\n");
        return 1;
    }

    char filename[100];
    int height, width, max_order, *dim_order, *device_dim_order;
    strcpy(filename, argv[1]);
    height = atoi(argv[2]);
    width = atoi(argv[3]);
    max_order = atoi(argv[4]);

    float kernelElapsedTime = 0;
    float kernelElapsedTime1 = 0;
    float kernelElapsedTime_poly = 0;
    float kernelElapsedTime_cheby_block_mom = 0;
    float kernelElapsedTime_block_moment = 0;
    float gr_time = 0;
    cudaEvent_t start, stop, start11, stop11, start_poly, stop_poly, start_block_moment, stop_block_moment, start_cheby_block_mom, stop_cheby_block_mom, gr_start, gr_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start11);
    cudaEventCreate(&stop11);
    cudaEventCreate(&start_poly);
    cudaEventCreate(&stop_poly);
    cudaEventCreate(&start_cheby_block_mom);
    cudaEventCreate(&stop_cheby_block_mom);
    cudaEventCreate(&gr_start);
    cudaEventCreate(&gr_stop);

    cudaEventCreate(&start_block_moment);
    cudaEventCreate(&stop_block_moment);
    struct timeval tim;
    double start1, stop1, start2, stop2, gr_start1, gr_stop1, gr_start2, gr_stop2, gr_start3, gr_stop3, gr_total1 = 0, gr_total2 = 0, gr_total3 = 0;
    int *host_interval_row, *device_interval_row, *host_check, *device_check, *device_blockno, *host_blockno, blockno = 0;
    unsigned char *host_image, *device_image, **blockimg, *bitplane, *device_bitplane;
    introw_t *device_ir;
    blocktype *host_block, *device_block, *host_a, *device_a;
    dim_order = (int *)malloc(2 * sizeof(int));
    dim_order[0] = MAX(height, width); // MAX
    dim_order[1] = max_order;

    size_t bytes = height * width * sizeof(unsigned char);

    float *host_poly, *device_poly, *reconstructed_image;
    float *device_ttx, *device_tb, *device_r, *host_r;
    float *device_stx, *device_sty, *host_tb, *host_t2d, *host_bmc, *host_bitplane_bmc, *device_bitplane_bmc, *host_ttx, *halfint_tb, *halfint_stx, *halfint_sty;
    blockimg = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (int i = 0; i < height; i++)
    {
        blockimg[i] = (unsigned char *)malloc(width * sizeof(int));
        if (!blockimg[i])
        {
            printf("not enough memory\n");
            exit(1);
        }
    }
    host_poly = (float *)malloc(dim_order[0] * dim_order[1] * sizeof(float));
    host_tb = (float *)calloc(dim_order[1] * dim_order[1], sizeof(float));
    host_bitplane_bmc = (float *)calloc(dim_order[1] * dim_order[1], sizeof(float));
    host_bmc = (float *)calloc(dim_order[1] * dim_order[1], sizeof(float));
    host_t2d = (float *)malloc(dim_order[1] * dim_order[1] * sizeof(float));
    host_blockno = (int *)malloc(height * (width / 2) * sizeof(int));
    reconstructed_image = (float *)malloc(height * width * sizeof(float));
    host_a = (blocktype *)malloc(height * width * sizeof(blocktype));
    host_check = (int *)malloc(height * width * sizeof(int));
    host_block = (blocktype *)malloc(height * width * sizeof(blocktype));
    host_image = (unsigned char *)malloc(bytes);
    bitplane = (unsigned char *)malloc(bytes);
    host_interval_row = (int *)malloc(height * sizeof(int));
    host_ttx = (float *)calloc(dim_order[1] * dim_order[0] + dim_order[0], sizeof(float));
    host_r = (float *)calloc(dim_order[1], sizeof(float));

    gpuErrchk(cudaMalloc(&device_dim_order, 2 * sizeof(int)));
    gpuErrchk(cudaMalloc(&device_a, height * width * sizeof(blocktype)));
    gpuErrchk(cudaMalloc(&device_blockno, height * (width / 2) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_poly, dim_order[0] * dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_ttx, (dim_order[1] * dim_order[0] + dim_order[0]) * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_r, dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_tb, dim_order[1] * dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_bitplane_bmc, dim_order[1] * dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc(&device_block, height * width * sizeof(blocktype)));
    gpuErrchk(cudaMalloc(&device_check, height * width * sizeof(int)));
    gpuErrchk(cudaMalloc(&device_ir, height * width * sizeof(introw_t)));
    gpuErrchk(cudaMalloc(&device_image, bytes));
    gpuErrchk(cudaMalloc(&device_bitplane, bytes));
    gpuErrchk(cudaMalloc(&device_interval_row, height * sizeof(int)));

    read(filename, host_image, height, width);

    gettimeofday(&tim, NULL);
    start1 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));
    gpuErrchk(cudaMemcpy(device_image, host_image, bytes, cudaMemcpyHostToDevice));
    gettimeofday(&tim, NULL);
    stop1 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

    gpuErrchk(cudaMemcpy(device_dim_order, dim_order, 2 * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_interval_row, host_interval_row, height * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start_cheby_block_mom);
    tt_calculation<<<GRID_SIZE, BLOCK_SIZE>>>(device_dim_order, device_ttx);
    cudaEventRecord(stop_cheby_block_mom);
    cudaEventSynchronize(stop_cheby_block_mom);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime_cheby_block_mom, start_cheby_block_mom, stop_cheby_block_mom));

    gpuErrchk(cudaEventRecord(start_poly));
    cheby_poly<<<GRID_SIZE, BLOCK_SIZE>>>(device_poly, device_dim_order);
    cudaEventRecord(stop_poly);
    cudaEventSynchronize(stop_poly);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime_poly, start_poly, stop_poly));

    int max_real = 4;
    for (int k = 0; k < max_real; k++)
    {

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                bitplane[i * width + j] = ((host_image[i * width + j] & (1 << (7 - k))) ? 1 : 0);

        gpuErrchk(cudaMemcpy(device_bitplane, bitplane, bytes, cudaMemcpyHostToDevice));

        gpuErrchk(cudaEventRecord(gr_start));

        pibr_extraction<<<GRID_SIZE, BLOCK_SIZE>>>(device_bitplane, height, width, device_interval_row, device_ir);
        pibr_block_creation<<<GRID_SIZE, BLOCK_SIZE>>>(device_ir, height, width, device_block, device_interval_row, device_check, device_blockno);
        pibr_block_creation2<<<GRID_SIZE, BLOCK_SIZE>>>(device_ir, height, width, device_block, device_interval_row, device_check, device_blockno);
        cudaEventRecord(gr_stop);
        cudaEventSynchronize(gr_stop);
        gpuErrchk(cudaEventElapsedTime(&gr_time, gr_start, gr_stop));

        gpuErrchk(cudaMemcpy(host_interval_row, device_interval_row, height * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(host_block, device_block, height * width * sizeof(blocktype), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(host_blockno, device_blockno, height * (width / 2) * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(host_check, device_check, height * width * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(host_poly, device_poly, dim_order[0] * dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));

        convert_block_indexing(host_block, host_check, height, width, host_a, host_interval_row, blockno, host_blockno);
        gpuErrchk(cudaMemcpy(device_a, host_a, height * width * sizeof(int), cudaMemcpyHostToDevice));

        int total = 0;
        for (int i = 0; i < height; i++)
        {
            total += host_interval_row[i];
        }

        write_poly(host_poly, dim_order);

        int total_blocks = 0;
        for (int i = 0; i < height; i++)
        {
            for (int y = 0; y < host_interval_row[i]; y++)
            {
                if (host_check[(i * height + y)] == 1)
                {
                    total_blocks++;
                }
            }
        }

        gpuErrchk(cudaMalloc((void **)&device_stx, dim_order[1] * total * sizeof(float)));
        gpuErrchk(cudaMalloc((void **)&device_sty, dim_order[1] * total * sizeof(float)));

        gettimeofday(&tim, NULL);
        gr_start2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

        tb_calculation<<<total_blocks, BLOCK_SIZE>>>(device_dim_order, device_a, device_ttx, device_stx, device_sty, total_blocks, device_bitplane_bmc, device_r);
        cudaDeviceSynchronize();
        gettimeofday(&tim, NULL);
        gr_stop2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

        gpuErrchk(cudaMemcpy(host_bitplane_bmc, device_bitplane_bmc, dim_order[1] * dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));

        gettimeofday(&tim, NULL);
        gr_start3 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

        for (int p = 0; p < dim_order[1]; p++)
            for (int q = 0; q < dim_order[1]; q++)
                host_bmc[q * dim_order[1] + p] += pow(2, 7 - k) * host_bitplane_bmc[q * dim_order[1] + p];

        gettimeofday(&tim, NULL);
        gr_stop3 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

        gr_total1 += gr_time;
        gr_total2 += (gr_stop2 - gr_start2);
        gr_total3 += (gr_stop3 - gr_start3);
    }

    float s;
    if (max_real == 2)
        s = 31.5;
    else if (max_real == 3)
        s = 15.5;
    else if (max_real == 4)
        s = 7.5;
    else if (max_real == 5)
        s = 3.5;
    else if (max_real == 6)
        s = 1.5;
    else if (max_real == 7)
        s = 0.5;
    else if (max_real == 8)
        s = 0;

    gpuErrchk(cudaMemcpy(host_ttx, device_ttx, dim_order[1] * dim_order[0] + dim_order[0] * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_r, device_r, dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));

    // half intensity iamge block computation
    halfint_tb = (float *)calloc(dim_order[1] * dim_order[1], sizeof(float));
    halfint_stx = (float *)calloc(dim_order[1], sizeof(float));
    halfint_sty = (float *)calloc(dim_order[1], sizeof(float));

    half_intensity_calc(3, dim_order, height, width, host_ttx, halfint_tb, halfint_stx, halfint_sty, host_r);

    for (int p = 0; p < dim_order[1]; p++)
        for (int q = 0; q < dim_order[1]; q++)
            host_bmc[q * dim_order[1] + p] += s * halfint_tb[q * dim_order[1] + p];

    gettimeofday(&tim, NULL);
    start2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));
    gpuErrchk(cudaMemcpy(host_tb, device_tb, dim_order[1] * dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));
    gettimeofday(&tim, NULL);
    stop2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

    //  cheby_rebuild(height, width, dim_order, reconstructed_image, host_poly, host_bmc);
    //  write1d(reconstructed_image, height, width);

    // make block image from extracted blocks
    // make_block_image(width, height, blockimg, host_block, host_check, host_interval_row);
    // write2d(blockimg, height, width);

    printf("data transfers time = %f msec\n\n", (stop1 - start1) + (stop2 - start2));
    printf("interval extraction = %1.4f msec\n", gr_total1);
    printf("poly processing time = %f msec\n", kernelElapsedTime_poly);
    printf("modified poly processing time = %f msec\n", kernelElapsedTime_cheby_block_mom);
    printf("BLOCK MOMENT processing time = %f msec\n", (gr_total2 + gr_total3));

    cudaFree(device_stx);
    cudaFree(device_sty);
    cudaFree(device_block);
    cudaFree(device_check);
    cudaFree(device_interval_row);
    cudaFree(device_ttx);
    cudaFree(device_dim_order);
    cudaFree(device_a);
    cudaFree(device_r);
    cudaFree(device_tb);
    free(host_interval_row);
    cudaFree(device_image);
    cudaFree(device_poly);
    cudaFree(device_ir);
    cudaFree(device_interval_row);
    free(host_image);
    free(host_check);
    free(host_poly);
    free(host_t2d);
    free(host_tb);
    free(dim_order);
    free(host_block);
    free(host_a);

    return 0;
}
