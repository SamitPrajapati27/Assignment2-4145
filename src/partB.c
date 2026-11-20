#include "assignment2.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void tiled_kernel(int M, int K, int N, int r0, int r1, int c0, int c1, int b, const double *A, const double *B, double *C) {
    if (b == 0) {
        for (int i = r0; i < r1; i++) {
            for (int j = c0; j < c1; j++) {
                C[i * N + j] = 0.0;
            }
        }
        for (int i = r0; i < r1; i++) {
            for (int k = 0; k < K; k++) {
                double a_ik = A[i * K + k];
                for (int j = c0; j < c1; j++) {
                    C[i * N + j] += a_ik * B[k * N + j];
                }
            }
        }
    } else {
        for (int i = r0; i < r1; i++) {
            for (int j = c0; j < c1; j++) {
                C[i * N + j] = 0.0;
            }
        }
        for (int ti = r0; ti < r1; ti += b) {
            int i_end = (ti + b > r1) ? r1 : ti + b;
            for (int tj = c0; tj < c1; tj += b) {
                int j_end = (tj + b > c1) ? c1 : tj + b;
                for (int tk = 0; tk < K; tk += b) {
                    int k_end = (tk + b > K) ? K : tk + b;
                    
                    for (int i = ti; i < i_end; i++) {
                        for (int k = tk; k < k_end; k++) {
                            double a_ik = A[i * K + k];
                            for (int j = tj; j < j_end; j++) {
                                C[i * N + j] += a_ik * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

int verify(int M, int N, const double *C, const double *C_ref) {
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_ref[i]) return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        printf("Usage: %s pr pc b M K N\n", argv[0]);
        return 1;
    }
    int pr = atoi(argv[1]), pc = atoi(argv[2]), b = atoi(argv[3]);
    int M = atoi(argv[4]), K = atoi(argv[5]), N = atoi(argv[6]);
    int p = pr * pc;
    if (p == 0) p = 1;

    double *A = generate_matrix(M, K, 1);
    double *B = generate_matrix(K, N, 2);
    double *C = (double*)malloc(M * N * sizeof(double));
    double *C_ref = (double*)malloc(M * N * sizeof(double));

    omp_set_num_threads(p);

    if (pr == 0 && pc == 0) {
        tiled_kernel(M, K, N, 0, M, 0, N, b, A, B, C);
    } else {
        #pragma omp parallel for schedule(static) default(none) shared(M, K, N, pr, pc, b, p, A, B, C)
        for (int rank = 0; rank < p; rank++) {
            int bi = rank / pc;
            int bj = rank % pc;
            
            int row_chunk = M / pr;
            int r0 = bi * row_chunk;
            int r1 = (bi == pr - 1) ? M : r0 + row_chunk;
            
            int col_chunk = N / pc;
            int c0 = bj * col_chunk;
            int c1 = (bj == pc - 1) ? N : c0 + col_chunk;
            
            tiled_kernel(M, K, N, r0, r1, c0, c1, b, A, B, C);
        }
    }

    tiled_kernel(M, K, N, 0, M, 0, N, 0, A, B, C_ref);
    if (!verify(M, N, C, C_ref)) {
        printf("Verification failed!\n");
        return 1;
    }

    int reps = 10;
    double times[10];
    for (int r = 0; r < reps; r++) {
        double start;
        if (pr == 0 && pc == 0) {
            start = omp_get_wtime();
            tiled_kernel(M, K, N, 0, M, 0, N, b, A, B, C);
            times[r] = omp_get_wtime() - start;
        } else {
            start = omp_get_wtime();
            #pragma omp parallel for schedule(static) default(none) shared(M, K, N, pr, pc, b, p, A, B, C)
            for (int rank = 0; rank < p; rank++) {
                int bi = rank / pc;
                int bj = rank % pc;
                
                int row_chunk = M / pr;
                int r0 = bi * row_chunk;
                int r1 = (bi == pr - 1) ? M : r0 + row_chunk;
                
                int col_chunk = N / pc;
                int c0 = bj * col_chunk;
                int c1 = (bj == pc - 1) ? N : c0 + col_chunk;
                
                tiled_kernel(M, K, N, r0, r1, c0, c1, b, A, B, C);
            }
            times[r] = omp_get_wtime() - start;
        }
    }

    for (int i = 0; i < reps - 1; i++) {
        for (int j = 0; j < reps - i - 1; j++) {
            if (times[j] > times[j+1]) {
                double temp = times[j]; times[j] = times[j+1]; times[j+1] = temp;
            }
        }
    }
    double median = (times[4] + times[5]) / 2.0;
    double min = times[0], max = times[reps-1], sum = 0;
    for (int i = 0; i < reps; i++) sum += times[i];
    
    double flops = 2.0 * M * K * N;
    double rate = flops / median;

    printf("Min: %f, Max: %f, Avg: %f, Median: %f, Rate: %e\n", min, max, (sum / reps), median, rate);

    free(A); free(B); free(C); free(C_ref);
    return 0;
}