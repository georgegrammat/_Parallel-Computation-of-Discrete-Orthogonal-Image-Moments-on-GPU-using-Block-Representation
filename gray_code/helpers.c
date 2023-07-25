#include "pibr.h"

void read(char *file, unsigned char *a, int image_height, int image_width)
{
    FILE *fname1;
    if ((fname1 = fopen(file, "rb")) == NULL)
    {
        printf("current frame doesn't exist\n");
        exit(-1);
    }
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            a[i * image_width + j] = fgetc(fname1);
        }
    }
    fclose(fname1);
}

// write blocks in a txt file, compare it with serial ibr
void write(int height, int *is_block, blocktype *block, int *host_interval_row, int *blockno)
{
    FILE *fname2;
    fname2 = fopen("blocks.txt", "wb");
    if (fname2)
    {
        for (int i = 0; i < height; i++)
        {
            for (int y = 0; y < blockno[i]; y++)
            {

                fprintf(fname2, "%d %d %d %d\n", block[(i * height + y)].x1, block[(i * height + y)].x2, block[(i * height + y)].y1, block[(i * height + y)].y2);
            }
        }
    }
    fclose(fname2);
}

void write_poly(float *poly, int *dim_order)
{
    FILE *fname2;
    fname2 = fopen("poly_par.txt", "wb");
    if (fname2)
    {
        for (int i = 0; i < dim_order[1] * dim_order[0]; i++)
        {
            fprintf(fname2, " %f \n", poly[i]);
        }
    }
}

void write1d(float *output_array, int image_height, int image_width)
{
    FILE *fname2;

    fname2 = fopen("gray_reb_img.raw", "wb");

    for (int l = 0; l < image_height; l++)
    {
        for (int k = 0; k < image_width; k++)
        {
            fputc(output_array[l * image_width + k], fname2);
        }
    }

    fclose(fname2);
}

void write2d(unsigned char **output_array, int image_height, int image_width)
{
    FILE *fname2;

    fname2 = fopen("blockimg.raw", "wb");

    for (int l = 0; l < image_height; l++)
    {
        for (int k = 0; k < image_width; k++)
        {
            fputc(output_array[l][k], fname2);
        }
    }

    fclose(fname2);
}

// function to convert block indexing to 0-total blocks
void convert_block_indexing(blocktype *block, int *is_block, int height, int width, blocktype *a, int *interval_row, int total, int *blockno)
{
    total = -1;
    for (int j = 0; j < height; j++)
    {
        for (int y = 0; y < blockno[j]; y++)
        {
            total++;
            a[total].x1 = block[(j * height + y)].x1;
            a[total].x2 = block[(j * height + y)].x2;
            a[total].y1 = block[(j * height + y)].y1;
            a[total].y2 = block[(j * height + y)].y2;
        }
    }
    // printf("total blocks (i) = %d\n", i);
}

void make_block_image(int width, int height, unsigned char **blockimg, blocktype *block, int *is_block, int *interval_row)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            blockimg[i][j] = 0;

    for (int k = 0; k < height; k++)
    {
        for (int y = 0; y < interval_row[k]; y++)
        {
            if (is_block[k * height + y])
            {
                for (int i = block[k * height + y].x1; i <= block[k * height + y].x2; i++)
                {
                    blockimg[block[k * height + y].y1][i] = 255;
                    blockimg[block[k * height + y].y2][i] = 255;
                }
                for (int j = block[k * height + y].y1; j <= block[k * height + y].y2; j++)
                {
                    blockimg[j][block[k * height + y].x1] = 255;
                    blockimg[j][block[k * height + y].x2] = 255;
                }
            }
        }
    }
}

void cheby_2d_mom(float *r, int *dim_order, float *tt, float *t2d, float *tb, unsigned char *img)
{
    long double error_moments;
    // int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    for (int x = 0; x < dim_order[0]; x++)
        for (int y = 0; y < dim_order[0]; y++)
            if (img[x * dim_order[0] + y] > 0)
                img[x * dim_order[0] + y] = 1;

    r[0] = dim_order[0];
    for (int n = 1; n < dim_order[1]; n++)
    {
        r[n] = r[n - 1] * (1 - n * n / (dim_order[0] * dim_order[0]));
    }
    for (int n = 0; n < dim_order[1]; n++)
    {
        r[n] /= (2 * n + 1);
        // printf("r[%d] = %f\n", n, host_r[n]);
    }

    // if(thread_id < dim_order[0])
    for (int p = 0; p < dim_order[1]; p++)
    {
        for (int q = 0; q < dim_order[1]; q++)
        {
            t2d[p * dim_order[1] + q] = 0;
            for (int x = 0; x < dim_order[0]; x++)
            {
                for (int y = 0; y < dim_order[0]; y++)
                    t2d[p * dim_order[1] + q] += tt[p * dim_order[0] + y] * tt[q * dim_order[0] + x] * img[x * dim_order[0] + y];
            }
            t2d[p * dim_order[1] + q] /= (r[p] * r[q]);
            // printf("t2d[%d]=%f\n", p*dim_order[1] + q, t2d[p*dim_order[1] + q]);

            // error between moments
            error_moments += ABS(tb[p * dim_order[1] + q] - t2d[p * dim_order[1] + q]);
        }
    }
}

void cheby_rebuild(int height, int width, int *dim_order, float *reb_img, float *tt, float *tb)
{
    float min_IR, max_IR;

    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < dim_order[0]; y++)
        {
            reb_img[x * width + y] = 0;

            for (int p = 0; p < dim_order[1]; p++)
            {
                for (int q = 0; q < dim_order[1]; q++)
                {
                    reb_img[x * width + y] += tb[p * dim_order[1] + q] * tt[p * dim_order[0] + y] * tt[q * dim_order[0] + x];
                    // printf("reb_img = %f\n", reb_img[x*dim_order[0] + y]);
                }
            }
        }
    }
}

void half_intensity_calc(int k, int *dim_order, int height, int width, float *tt, float *tb, float *stx, float *sty, float *r)
{

    for (int p = 0; p < dim_order[1]; p++)
    {
        for (int x = 0; x < height; x++)
        {
            stx[p] += tt[p * dim_order[0] + x];
        }
    }

    for (int q = 0; q < dim_order[1]; q++)
    {
        for (int y = 0; y < width; y++)
        {
            sty[q] += tt[q * dim_order[0] + y];
        }
    }

    for (int p = 0; p < dim_order[1]; p++)
    {
        for (int q = 0; q < dim_order[1]; q++)
        {
            tb[p * dim_order[1] + q] = stx[p] * sty[q] / (r[p] * r[q]);
        }
    }
}

double nire_calculation(unsigned char *original_image, float *reconstructed_image, int height, int width)
{
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            sum1 += abs(original_image[i * height + j] - reconstructed_image[i * height + j]);
            sum2 += original_image[i * height + j] * original_image[i * height + j];
        }
    }

    return sum1 / sum2;
}