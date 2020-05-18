#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define size 2000
#define DATA float

DATA a[size] __attribute__((aligned(16)));
DATA b[size][size] __attribute__((aligned(16)));
DATA b2[size][size] __attribute__((aligned(16)));

DATA c[size][size] __attribute__((aligned(16)));
DATA c2[size][size] __attribute__((aligned(16)));
DATA out[size] __attribute__((aligned(16)));
DATA out2[size] __attribute__((aligned(16)));

DATA zero[4] __attribute__((aligned(16))) = {0, 0, 0, 0};


void generate_matrices()
{
    for (int i = 0; i < size; i++)
    {
        a[i] = ((i * 3 + 2) % 9) + 1;
    }

    for (int i = 0; i < size; i++)
    {
        for (int ii = 0; ii < size; ii++)
        {
            b[i][ii] = ((ii + i * 454 + 12) % 9) + 1;
            b2[i][ii] = ((ii + i * 44 + 2) % 9) + 1;
        }
    }

    for (int i = 0; i < size; i++)
    {
        out[i] = 0;
        out2[i] = 0;
    }
}

double seconds()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec / 1000000000.0;
}

static bool equals()
{
    for (int i = 0; i < size; i++)
    {
        if (out[i] != out2[i])
        {
            printf("%f  %f", out2[i], out[i]);
            return false;
        }
    }
    return true;
}

static void Matrix_Vector_simple()
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            out[i] += b[i][j] * a[j];
        }
    }
}

static void Matrix_Vector_simd()
{
    DATA prod = 0;
    __m128 X, Y, Z;
    for (int i = 0; i < size; i = i + 1)
    {
        Z = _mm_load_ps(&zero[0]);
        prod = 0;
        for (int j = 0; j < size; j = j + 4)
        {
            X = _mm_load_ps(&b[i][j]);
            Y = _mm_load_ps(&a[j]);
            X = _mm_mul_ps(X, Y);
            Z = _mm_add_ps(X, Z);
        }

        for (int ii = 0; ii < 4; ii++)
            prod += Z[ii];

        out2[i] = prod;
    }
}

static void mat_mat()
{
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            for (int j = 0; j < size; j++)
            {
                c[i][j] += b[i][k] * b2[k][j];
            }
        }
    }
}

static void mat_mat_simd()
{

  for (int i = 0; i < size; i++) {
    for (int ii = 0; ii < size; ii += 4) {
        __m128 sum =  _mm_load_ps(&zero[0]);

        for (int iii = 0; iii < size; iii++) {

            __m128 entry = _mm_set1_ps(b[i][iii]);
            __m128 row  = _mm_load_ps(&b2[iii][ii]);
            sum = _mm_add_ps(sum, _mm_mul_ps(entry, row));

        }
        _mm_store_ps(&c2[i][ii], sum);

    }
}
    return ;
}


static bool Mat_Mat_eq()
{

    for (int i = 0; i < size; i++)
    {
        for (int ii = 0; ii < size; ii++)
        {
            if (c[i][ii] != c2[i][ii])
            {
                return false;
                printf("%f \n", c[i][ii]);
                printf("%f \n", c2[i][ii]);
            }
        }
    }
    return true;
}


static void PrintMatrix(DATA m[size][size])
{
    printf("------------------------------------------------------------- \n");
    for (int i = 0; i < size; i++)
    {
        for (int ii = 0; ii < size; ii++)
        {
            printf("| %.1f ", m[i][ii]);
        }

        printf("| \n");
    }

    printf("------------------------------------------------------------ \n");
}

int main()
{
    double before, after;

    generate_matrices();

    before = seconds();
    mat_mat();
    after = seconds();
    printf("Time for Naive Matrix-Matrix of %d x %d : %f\n", size, size, after - before);

    before = seconds();
    mat_mat_simd();
    after = seconds();
    
    printf("Time for SMID Matrix-Matrix of %d x %d : %f\n", size, size, after - before);

    if(Mat_Mat_eq())
    printf("successful \n------------------- \n" );

    // PrintMatrix(c);

    before = seconds();

    Matrix_Vector_simple();

    after = seconds();

    printf("Time for Naive Matrix Vector   : %f\n", after - before);

    before = seconds();

    Matrix_Vector_simd();

    after = seconds();

    printf("Time for paralel Matrix Vector : %f\n", after - before);

    if (equals())
    {
        printf("successful \n");
    }

    /*

    for (int i = 0; i < size; i++)
    {
        printf("%f ", out[i]);
    }
    printf("\n");
    for (int i = 0; i < size; i++)
    {
        printf("%f ", out2[i]);
    }

    */

    return 0;
}