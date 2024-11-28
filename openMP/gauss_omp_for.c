#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

void prt1a(char *t1, float *v, int n, char *t2);
void wtime(double *t) {
    static int sec = -1;
    struct timeval tv;
    gettimeofday(&tv, (void *)0);
    if (sec < 0) sec = tv.tv_sec;
    *t = (tv.tv_sec - sec) + 1.0e-6 * tv.tv_usec;
}

int N;
float *A;
#define A(i, j) A[(i) * (N + 1) + (j)]
float *X;

int main(int argc, char **argv) {
    const char* omp_threads = getenv("OMP_NUM_THREADS");


    //double time0, time1;
    FILE *in;
    int i, j, k;

    in = fopen("data.in", "r");
    if (in == NULL) {
        printf("Cannot open 'data.in'\n");
        exit(1);
    }
    i = fscanf(in, "%d", &N);
    if (i < 1) {
        printf("Wrong 'data.in' format (N ...)\n");
        exit(2);
    }

    /* Create arrays */
    A = (float *)malloc(N * (N + 1) * sizeof(float));
    X = (float *)malloc(N * sizeof(float));

    printf("\n");
    printf("GAUSS %dx%d THREAD=%s\n----------------------------------\n", N, N, omp_threads);

    /* Initialize array A */
    #pragma omp parallel for private(i, j) shared(A, N)
    for (i = 0; i <= N - 1; i++) {
        for (j = 0; j <= N; j++) {
            if (i == j || j == N)
                A(i, j) = 1.f;
            else
                A(i, j) = 0.f;
        }
    }

    //wtime(&time0);
    double time_start = omp_get_wtime();

    /* Elimination */
    for (i = 0; i < N - 1; i++) {
        #pragma omp parallel for private(k, j) shared(A, i, N)
        for (k = i + 1; k < N; k++) {
            for (j = i + 1; j <= N; j++) {
                A(k, j) -= A(k, i) * A(i, j) / A(i, i);
            }
        }
    }

    /* Reverse substitution */
    X[N - 1] = A(N - 1, N) / A(N - 1, N - 1);

    for (j = N - 2; j >= 0; j--) {
        #pragma omp parallel for private(k) shared(A, X, j, N)
        for (k = 0; k <= j; k++) {
            A(k, N) -= A(k, j + 1) * X[j + 1];
        }
        X[j] = A(j, N) / A(j, j);
    }

    //wtime(&time1);
    double time_end = omp_get_wtime();
    double delta = time_end - time_start;
    printf("%0.6lf\n", delta);
    //printf("Time in seconds = %g s\n", time1 - time0);
    prt1a("X = (", X, N > 9 ? 9 : N, "...)\n");

    free(A);
    free(X);
    return 0;
}

void prt1a(char *t1, float *v, int n, char *t2) {
    int j;
    printf("%s", t1);
    for (j = 0; j < n; j++)
        printf("%.4g%s", v[j], (j % 10 == 9) ? "\n" : ", ");
    printf("%s", t2);
}
