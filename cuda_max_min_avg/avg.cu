#include <stdio.h>
#include <stdlib.h>

#define N 100

__global__ void max_kernel(int *a, int *n) {
    int tid = threadIdx.x;
    int start = *n * tid;
    int end = *n * (tid + 1);
    
    for (int i = start; i < end; i++) {    
            a[start] += a[i];
    }
}

int main() {
    
    srand(time(NULL));
	int *arr, *d_arr, *dev_n;
	int n = 5;   
	arr = new int[N];
	
   	for (int i = 0; i < N; i++) {
        arr[i] = rand() % 1000;
	}
	
	cudaMalloc(&d_arr, N * sizeof(int));
	cudaMalloc(&dev_n, sizeof(int));
	
	cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	
	int len = N;
    
    while (len > n) {
    
        max_kernel<<<1, len / n>>>(d_arr, dev_n);
        
        cudaDeviceSynchronize();
        
        cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Pass %d: ", pass++);
        for (int i = 0; i < (len/n); i++) {
            arr[i] = arr[i * n];
            printf("%d ", arr[i]);
        }
        printf("\n");
        
    	cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);
    	
        len /= n;
    }
    
    for (int i = 0; i < len; i++) {        
            arr[0] += arr[i];
    }
    
    printf("average : %d\n", arr[0] / 100);
    
    return 0;
}
