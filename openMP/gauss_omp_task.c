#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>


void prt1a(char *t1, float *v, int n,char *t2);
void wtime(double *t) {
	static int sec = -1;
	struct timeval tv;
	gettimeofday(&tv, (void *)0);
	if (sec < 0) sec = tv.tv_sec;
	*t = (tv.tv_sec - sec) + 1.0e-6*tv.tv_usec;
}

int N;
float *A;
#define A(i,j) A[(i)*(N+1)+(j)]
#define tasksize 32
#define min(a, b) (a) < (b) ? (a) : (b)
float *X;


void elimination_phase(float *A, int i, int k_start)
{
    int k_end = min(k_start + tasksize, N);
    for (int k = k_start; k < k_end; k++)
    {
        for (int j = i + 1; j <= N; j++)
            A(k, j) = A(k, j) - A(k, i) * A(i, j) / A(i, i);
    }
}

void reverse_phase(float *A, float *X, int j, int k_start) {
    int k_end = min(k_start + tasksize, j+1);
    for (int k = k_start; k < k_end; k++)
    {
        A(k, N) = A(k, N) - A(k, j + 1) * X[j + 1];
    }
}


int main(int argc,char **argv) {
	const char* omp_threads = getenv("OMP_NUM_THREADS");
    int threads = 1;
    
    if (omp_threads != NULL) {
        threads = atoi(omp_threads);
    }

	//double time0, time1;
	FILE *in;
	int i, j, k;
	in=fopen("data.in","r");
	if(in==NULL) {
		printf("Can not open 'data.in' "); exit(1);
	}
	i=fscanf(in,"%d", &N);
	if(i<1) {
		printf("Wrong 'data.in' (N ...)"); exit(2);
	}
	/* create arrays */
	A=(float *)malloc(N*(N+1)*sizeof(float));
	X=(float *)malloc(N*sizeof(float));
	
	printf("\n");
    printf("GAUSS %dx%d THREAD=%s\n----------------------------------\n", N, N, omp_threads);	
	
	/* initialize array A*/
	for(i=0; i <= N-1; i++)
		for(j=0; j <= N; j++)
			if (i==j || j==N)
				A(i,j) = 1.f;
			else 
				A(i,j)=0.f;
	
	//wtime(&time0);
	double time_start = omp_get_wtime();

	/* elimination */
	for (i=0; i<N-1; i++) {
		#pragma omp parallel shared(i) private(j, k)
		#pragma omp single
		for (k = i + 1; k <= N - 1; k += tasksize)
			#pragma omp task shared(A) shared(i, k)
        	{
            	elimination_phase(A, i, k);
       		}
	}


	/* reverse substitution */
    X[N - 1] = A(N - 1, N) / A(N - 1, N - 1);
    for (j = N - 2; j >= 0; j--)
    {
        #pragma omp parallel shared(A,j) private(k)
        for (k = 0; k <= j; k += tasksize)
        {
            #pragma omp task firstprivate(k)
            {
                reverse_phase(A,X,j,k);
            }
        }
    }
    for (j = N - 2; j >= 0; j--)
    {
        X[j] = A(j, N) / A(j, j);
    }

	//wtime(&time1);
	//printf("Time in seconds=%gs\n",time1-time0);
	
	double time_end = omp_get_wtime();
    double delta = time_end - time_start;
    printf("%0.6lf\n", delta);
	
	prt1a("X=(", X,N>9?9:N,"...)\n");
	free(A);
	free(X);
	return 0;
}

void prt1a(char * t1, float *v, int n,char *t2) {
	int j;
	printf("%s",t1);
	for(j=0;j<n;j++)
		printf("%.4g%s",v[j], j%10==9? "\n": ", ");
	printf("%s",t2);
}