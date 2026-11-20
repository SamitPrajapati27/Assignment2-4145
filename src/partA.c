#include "assignment2.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int rank;
    int pr;
    int pc;
    int M;
    int K;
    int N;
    int b;
    const double *A;
    const double *B;
    double *C;
} thread_arg_t;

static void tiled_kernel(int K, int N, int r0, int r1, int c0, int c1, int b,
                         const double *A, const double *B, double *C) {
    if (b == 0) {
        for (int i = r0; i < r1; i++) {
            for (int j = c0; j < c1; j++) {
                C[(size_t)i * N + j] = 0.0;
            }
        }
        for (int i = r0; i < r1; i++) {
            for (int k = 0; k < K; k++) {
                double a_ik = A[(size_t)i * K + k];
                for (int j = c0; j < c1; j++) {
                    C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
                }
            }
        }
    } else {
        for (int ti = r0; ti < r1; ti += b) {
            int i_end = (ti + b < r1) ? (ti + b) : r1;
            for (int tj = c0; tj < c1; tj += b) {
                int j_end = (tj + b < c1) ? (tj + b) : c1;

                for (int i = ti; i < i_end; i++) {
                    for (int j = tj; j < j_end; j++) {
                        C[(size_t)i * N + j] = 0.0;
                    }
                }

                for (int tk = 0; tk < K; tk += b) {
                    int k_end = (tk + b < K) ? (tk + b) : K;
                    for (int i = ti; i < i_end; i++) {
                        for (int k = tk; k < k_end; k++) {
                            double a_ik = A[(size_t)i * K + k];
                            for (int j = tj; j < j_end; j++) {
                                C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void *thread_worker(void *arg) {
    const thread_arg_t *t = (const thread_arg_t *)arg;

    int bi = t->rank / t->pc;
    int bj = t->rank % t->pc;

    int row_chunk = t->M / t->pr;
    int r0 = bi * row_chunk;
    int r1 = (bi == t->pr - 1) ? t->M : (r0 + row_chunk);

    int col_chunk = t->N / t->pc;
    int c0 = bj * col_chunk;
    int c1 = (bj == t->pc - 1) ? t->N : (c0 + col_chunk);

    tiled_kernel(t->K, t->N, r0, r1, c0, c1, t->b, t->A, t->B, t->C);
    return NULL;
}

static int verify_exact(int M, int N, const double *C, const double *C_ref) {
    size_t total = (size_t)M * (size_t)N;
    for (size_t idx = 0; idx < total; idx++) {
        if (C[idx] != C_ref[idx]) {
            return 0;
        }
    }
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s pr pc b M K N [reps]\n", argv[0]);
        return 1;
    }

    int pr = atoi(argv[1]);
    int pc = atoi(argv[2]);
    int b  = atoi(argv[3]);
    int M  = atoi(argv[4]);
    int K  = atoi(argv[5]);
    int N  = atoi(argv[6]);
    int reps = (argc >= 8) ? atoi(argv[7]) : 10;

    int is_seq = (pr == 0 && pc == 0);
    int p = is_seq ? 0 : (pr * pc);

    double *A = generate_matrix(M, K, 1);
    double *B = generate_matrix(K, N, 2);
    double *C = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
    double *C_ref = (double *)malloc((size_t)M * (size_t)N * sizeof(double));
    double *times = (double *)malloc((size_t)reps * sizeof(double));
    if (!C || !C_ref || !times) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    pthread_t *threads = NULL;
    thread_arg_t *args = NULL;
    if (!is_seq) {
        threads = (pthread_t *)malloc((size_t)p * sizeof(pthread_t));
        args = (thread_arg_t *)malloc((size_t)p * sizeof(thread_arg_t));
        if (!threads || !args) {
            fprintf(stderr, "Out of memory\n");
            return 1;
        }
        for (int i = 0; i < p; i++) {
            args[i].rank = i;
            args[i].pr = pr;
            args[i].pc = pc;
            args[i].M = M;
            args[i].K = K;
            args[i].N = N;
            args[i].b = b;
            args[i].A = A;
            args[i].B = B;
            args[i].C = C;
        }
    }

    tiled_kernel(K, N, 0, M, 0, N, 0, A, B, C_ref);

    if (is_seq) {
        tiled_kernel(K, N, 0, M, 0, N, b, A, B, C);
    } else {
        for (int i = 0; i < p; i++) {
            pthread_create(&threads[i], NULL, thread_worker, &args[i]);
        }
        for (int i = 0; i < p; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    int ok = verify_exact(M, N, C, C_ref);
    if (!ok) {
        fprintf(stderr, "VERIFICATION FAILED for pr=%d pc=%d b=%d M=%d K=%d N=%d\n",
                pr, pc, b, M, K, N);
        return 1;
    }

    for (int r = 0; r < reps; r++) {
        double t0 = now_seconds();
        if (is_seq) {
            tiled_kernel(K, N, 0, M, 0, N, b, A, B, C);
        } else {
            for (int i = 0; i < p; i++) {
                pthread_create(&threads[i], NULL, thread_worker, &args[i]);
            }
            for (int i = 0; i < p; i++) {
                pthread_join(threads[i], NULL);
            }
        }
        times[r] = now_seconds() - t0;
    }

    qsort(times, (size_t)reps, sizeof(double), cmp_double);
    double min_t = times[0];
    double max_t = times[reps - 1];
    double median_t = (reps % 2 == 0)
                          ? 0.5 * (times[reps / 2 - 1] + times[reps / 2])
                          : times[reps / 2];
    double sum_t = 0.0;
    for (int r = 0; r < reps; r++) {
        sum_t += times[r];
    }
    double avg_t = sum_t / (double)reps;
    double flops = 2.0 * (double)M * (double)K * (double)N;
    double rate = flops / median_t;

    printf("partA,%d,%d,%d,%d,%d,%d,%d,PASS,%.6f,%.6f,%.6f,%.6f,%.6e\n",
           pr, pc, p, b, M, K, N, min_t, median_t, avg_t, max_t, rate);

    free(threads);
    free(args);
    free(times);
    free(C_ref);
    free(C);
    free(B);
    free(A);
    return 0;
}