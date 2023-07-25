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
    float kernelElapsedTime11 = 0;
    float kernelElapsedTime_poly = 0;
    float kernelElapsedTime_cheby_block_mom = 0;
    float kernelElapsedTime_block_moment = 0;
    cudaEvent_t start, stop, start11, stop11, start111, stop111, start_poly, stop_poly, start_block_moment, stop_block_moment, start_cheby_block_mom, stop_cheby_block_mom;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start11);
    cudaEventCreate(&stop11);
    cudaEventCreate(&start111);
    cudaEventCreate(&stop111);
    cudaEventCreate(&start_poly);
    cudaEventCreate(&stop_poly);
    cudaEventCreate(&start_cheby_block_mom);
    cudaEventCreate(&stop_cheby_block_mom);

    cudaEventCreate(&start_block_moment);
    cudaEventCreate(&stop_block_moment);
    struct timeval tim;
    double start1, stop1, start2, stop2;
    int *host_interval_row, *device_interval_row, *host_check, *device_check, *device_blockno, *host_blockno, blockno = 0;
    unsigned char *host_image, *device_image, **blockimg;
    introw_t *device_ir;
    blocktype *host_block, *device_block, *host_a, *device_a;
    dim_order = (int *)malloc(2 * sizeof(int));
    dim_order[0] = MAX(height, width); // MAX
    dim_order[1] = max_order;

    size_t bytes = height * width * sizeof(unsigned char);

    float *host_poly, *device_poly, *reconstructed_image;
    float *device_ttx, *device_tb, *device_r;
    float *device_stx, *device_sty, *host_tb, *host_t2d;

    host_poly = (float *)malloc(dim_order[0] * dim_order[1] * sizeof(float));
    host_tb = (float *)calloc(dim_order[1] * dim_order[1], sizeof(float));
    host_t2d = (float *)malloc(dim_order[1] * dim_order[1] * sizeof(float));
    host_blockno = (int *)malloc(height * (width / 2) * sizeof(int));
    reconstructed_image = (float *)malloc(dim_order[0] * dim_order[0] * sizeof(float));
    host_a = (blocktype *)malloc(height * width * sizeof(blocktype));
    host_check = (int *)malloc(height * width * sizeof(int));
    host_block = (blocktype *)malloc(height * width * sizeof(blocktype));
    host_image = (unsigned char *)malloc(bytes);
    host_interval_row = (int *)malloc(height * sizeof(int));

    gpuErrchk(cudaMalloc(&device_dim_order, 2 * sizeof(int)));
    gpuErrchk(cudaMalloc(&device_a, height * width * sizeof(blocktype)));
    gpuErrchk(cudaMalloc(&device_blockno, height * (width / 2) * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_poly, dim_order[0] * dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_ttx, (dim_order[1] * dim_order[0] + dim_order[0]) * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_r, dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&device_tb, dim_order[1] * dim_order[1] * sizeof(float)));
    gpuErrchk(cudaMalloc(&device_block, height * width * sizeof(blocktype)));
    gpuErrchk(cudaMalloc(&device_check, height * width * sizeof(int)));
    gpuErrchk(cudaMalloc(&device_ir, height * width * sizeof(introw_t)));
    gpuErrchk(cudaMalloc(&device_image, bytes));
    gpuErrchk(cudaMalloc(&device_interval_row, height * sizeof(int)));

    read(filename, host_image, height, width);

    gettimeofday(&tim, NULL);
    start1 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));
    gpuErrchk(cudaMemcpy(device_image, host_image, bytes, cudaMemcpyHostToDevice));
    gettimeofday(&tim, NULL);
    stop1 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

    gpuErrchk(cudaMemcpy(device_dim_order, dim_order, 2 * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_interval_row, host_interval_row, height * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaEventRecord(start));
    pibr_extraction<<<GRID_SIZE, BLOCK_SIZE>>>(device_image, height, width, device_interval_row, device_ir);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime, start, stop));

    cudaFree(device_image);

    gpuErrchk(cudaEventRecord(start11));
    pibr_block_creation<<<GRID_SIZE, BLOCK_SIZE>>>(device_ir, height, width, device_block, device_interval_row, device_check, device_blockno);
    cudaEventRecord(stop11);
    cudaEventSynchronize(stop11);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime1, start11, stop11));

    gpuErrchk(cudaEventRecord(start111));
    pibr_block_creation2<<<GRID_SIZE, BLOCK_SIZE>>>(device_ir, height, width, device_block, device_interval_row, device_check, device_blockno);
    cudaEventRecord(stop111);
    cudaEventSynchronize(stop111);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime11, start111, stop111));

    gpuErrchk(cudaEventRecord(start_poly));
    cheby_poly<<<GRID_SIZE, BLOCK_SIZE>>>(device_poly, device_dim_order);
    cudaEventRecord(stop_poly);
    cudaEventSynchronize(stop_poly);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime_poly, start_poly, stop_poly));

    gpuErrchk(cudaMemcpy(host_interval_row, device_interval_row, height * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_block, device_block, height * width * sizeof(blocktype), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_blockno, device_blockno, height * (width / 2) * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_check, device_check, height * width * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_poly, device_poly, dim_order[0] * dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(device_block);
    cudaFree(device_check);
    cudaFree(device_poly);
    cudaFree(device_ir);
    cudaFree(device_interval_row);

    write(dim_order[0], 0, host_block, host_interval_row, host_blockno);

    convert_block_indexing(host_block, host_check, height, width, host_a, host_interval_row, blockno, host_blockno);
    gpuErrchk(cudaMemcpy(device_a, host_a, height * width * sizeof(int), cudaMemcpyHostToDevice));

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

    cudaEventRecord(start_cheby_block_mom);
    tt_calculation<<<GRID_SIZE, BLOCK_SIZE>>>(device_dim_order, device_a, device_ttx, device_stx, device_sty);
    cudaEventRecord(stop_cheby_block_mom);
    cudaEventSynchronize(stop_cheby_block_mom);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime_cheby_block_mom, start_cheby_block_mom, stop_cheby_block_mom));

    cudaEventRecord(start_block_moment);
    tb_calculation<<<total_blocks, BLOCK_SIZE>>>(device_dim_order, device_a, device_ttx, device_stx, device_sty, total_blocks, device_tb, device_r);
    cudaEventRecord(stop_block_moment);
    cudaEventSynchronize(stop_block_moment);
    gpuErrchk(cudaEventElapsedTime(&kernelElapsedTime_block_moment, start_block_moment, stop_block_moment));

    gettimeofday(&tim, NULL);
    start2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));
    gpuErrchk(cudaMemcpy(host_tb, device_tb, dim_order[1] * dim_order[1] * sizeof(float), cudaMemcpyDeviceToHost));
    gettimeofday(&tim, NULL);
    stop2 = (tim.tv_sec * 1000.0 + (tim.tv_usec / 1000.0));

    // cheby_rebuild(height, width, dim_order, reconstructed_image, host_poly, host_tb);
    // write1d(reconstructed_image, height, width);

    double nire = nire_calculation(host_image, reconstructed_image, height, width);
    // make block image from extracted blocks
    // make_block_image(width, height, blockimg, host_block, host_check, host_interval_row);
    // write2d(blockimg, height, width);

    printf("total blocks %d\n", total_blocks);
    printf("intervals = %d\n\n", total);
    printf("data transfers time = %f msec\n\n", (stop1 - start1) + (stop2 - start2));
    printf("interval extraction = %1.4f msec\n", kernelElapsedTime);
    printf("interval matching = %1.4f msec\n\n", kernelElapsedTime1);
    printf("block extraction = %1.4f msec\n\n", kernelElapsedTime11);
    printf("poly processing time = %f msec\n", kernelElapsedTime_poly);
    printf("modified poly processing time = %f msec\n", kernelElapsedTime_cheby_block_mom);
    printf("BLOCK MOMENT processing time = %f msec\n", kernelElapsedTime_block_moment + kernelElapsedTime_cheby_block_mom);
    printf("NIRE = %f \n", nire);

    cudaFree(device_stx);
    cudaFree(device_sty);
    cudaFree(device_ttx);
    cudaFree(device_dim_order);
    cudaFree(device_a);
    cudaFree(device_r);
    cudaFree(device_tb);
    free(host_interval_row);
    free(host_check);
    free(host_poly);
    free(host_image);
    free(host_t2d);
    free(host_tb);
    free(dim_order);
    free(host_block);
    free(host_a);

    return 0;
}
