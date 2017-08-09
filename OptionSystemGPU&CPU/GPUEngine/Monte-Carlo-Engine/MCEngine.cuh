
#include "MCNeuron.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define imin(a,b) (a<b?a:b)
#include <chrono>

const int blockSize = 256;

__global__ void RunSelf(int N_, MCNeuron *d_a,int seed1=0) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	curandStatus_t * d_state;
	//todo:
	//random generator1: init every curandState partially in each neuron
	//can be optimized by initing curandState globally
	//see RunSelf2
	int seed = tid + seed1;
	if (tid < N_) {
		d_a[tid].doOneTrail(seed);
	}
}
__global__ void RunSelf2(int N_, MCNeuron *d_a,int seed1, curandState_t *d_state) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed1, tid, 0, &d_state[tid]);
	//__syncthreads();
	
	//todo
	//For some weild reason, __syncthreads() do not work on my PC
	if (tid < N_) {
		d_a[tid].doOneTrail2(d_state[tid]);
	}
}
__global__ void RunCallOptionPrice(int N_, MCNeuron *d_a, float strike, long double *partial_sums, double barrier = 0) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	//Intent:
	//switch the price computation into device by reduction strategy
	//however, for some strange reason, __syncthreads() is undefined on my Visual Studio
	//Anyway, it work, that's bizarre


	__shared__ double cache[blockSize];
	int cacheIndex = threadIdx.x;

	if (tid < N_) {
		cache[cacheIndex] = d_a[tid].upAndOutCallPayOff(strike, barrier);
	}
	//__syncthreads();	//undefined??
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		//_syncthreads();
		i = i / 2;
	}
	if (cacheIndex == 0)
		partial_sums[blockIdx.x] = cache[0];
	//__syncthreads();
	if (blockIdx.x == 0 && cacheIndex == 0) {
		long double price;
		int blocksPerGrid = (N_ + blockSize - 1) / blockSize;
		for (int i = 0; i < blocksPerGrid; ++i) {
			price += partial_sums[i];
		}
		price = price / N_;
		partial_sums[0] = price;
	}
}
__global__ void RunVanillaCallOptionPrice(int N_, MCNeuron *d_a, float strike, long double *partial_sums) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	//Intent:
	//switch the price computation into device by reduction strategy
	//however, for some strange reason, __syncthreads() is undefined on my Visual Studio
	//Anyway, it work, that's bizarre


	__shared__ double cache[blockSize];
	int cacheIndex = threadIdx.x;

	if (tid < N_) {
		cache[cacheIndex] = d_a[tid].vanillaCallPayOff(strike);
	}
	//__syncthreads();	//undefined??
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		//_syncthreads();
		i = i / 2;
	}
	if (cacheIndex == 0)
		partial_sums[blockIdx.x] = cache[0];
	//__syncthreads();
	if (blockIdx.x == 0 && cacheIndex == 0) {
		long double price;
		int blocksPerGrid = (N_ + blockSize - 1) / blockSize;
		for (int i = 0; i < blocksPerGrid; ++i) {
			price += partial_sums[i];
		}
		price = price / N_;
		partial_sums[0] = price;
	}
}
__global__ void RunAsianCallOptionPrice(int N_, MCNeuron *d_a, float strike,long double *partial_sums) {
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	//Intent:
	//switch the price computation into device by reduction strategy
	//however, for some strange reason, __syncthreads() is undefined on my Visual Studio
	//Anyway, it work, that's bizarre

	
	__shared__ double cache[blockSize];
	int cacheIndex = threadIdx.x;
	
	if (tid < N_) {
		cache[cacheIndex] = d_a[tid].asianCallPayOff(strike);
	}
	//__syncthreads();	//undefined??
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		//_syncthreads();
		i = i / 2;
	}
	if (cacheIndex == 0)
		partial_sums[blockIdx.x] = cache[0];
	//__syncthreads();
	if (blockIdx.x == 0 && cacheIndex == 0) {
		long double price;
		int blocksPerGrid = (N_ + blockSize - 1) / blockSize;
		for (int i = 0; i < blocksPerGrid; ++i) {
			price += partial_sums[i];
		}
		price = price / N_;
		partial_sums[0] = price;
	}
}

class MCEngine {
private:
	MCNeuron *d_a;
	MCNeuron *h_a;
	long double *d_prices;
	curandState_t *d_state;
	float spot;
	float r;
	float vol;
	float timeToMaturity;
	unsigned int n_steps;
	int N;
public:
	__host__ MCEngine(int n,double spot_=20,double r_=0.02,double vol_=0.2,double timeToMaturity_=1,unsigned int n_steps=300) :
		N(n),
		spot(spot_),
		r(r_),
		vol(vol_),
		timeToMaturity(timeToMaturity_),
		n_steps(n_steps){
		h_a = new MCNeuron[n];
		cudaMalloc((void**)&d_a, n * sizeof(MCNeuron));
		cudaMalloc((void**)&d_prices, n * sizeof(long double));
		for (int i = 1; i < n; ++i) {
			h_a[i] = 20;
		}
		cudaMemcpy(d_a, h_a, n * sizeof(MCNeuron), cudaMemcpyHostToDevice);
	}
	__host__ void doSimulation() {
		RunSelf << <(N + blockSize - 1) / blockSize, blockSize >> > (N, d_a,time(nullptr));
		cudaMemcpy(h_a, d_a, N * sizeof(MCNeuron), cudaMemcpyDeviceToHost);
	}
	__host__ void doSimulation2() {
		RunSelf2 << <(N + blockSize - 1) / blockSize, blockSize >> > (N, d_a, time(nullptr),d_state);
		cudaMemcpy(h_a, d_a, N * sizeof(MCNeuron), cudaMemcpyDeviceToHost);
	}

	__host__ void printNet() {
		for (int i = 0; i < N; ++i) {
			h_a[i].printX();
		}
	}
	__host__ double vanillaCallPrice(double strike) {
		double price=0;
		for (int i = 0; i < N; ++i) {
			price += h_a[i].vanillaCallPayOff(strike);
		}
		price = price / N;
		return price;
	}
	__host__ double vanillaCallPrice2(double strike) {
		RunVanillaCallOptionPrice <<<(N + blockSize - 1) / blockSize, blockSize >>>(N, d_a, strike, d_prices);
		double price;
		//cudaMemcpy(h_a, d_a, N * sizeof(MCNeuron), cudaMemcpyDeviceToHost);
		cudaMemcpy(&price, &d_prices[0],sizeof(double), cudaMemcpyDeviceToHost);
		return price;
	}
	__host__ double asianCallPrice2(double strike) {
		RunAsianCallOptionPrice << <(N + blockSize - 1) / blockSize, blockSize >> >(N, d_a, strike, d_prices);
		double price;
		//cudaMemcpy(h_a, d_a, N * sizeof(MCNeuron), cudaMemcpyDeviceToHost);
		cudaMemcpy(&price, &d_prices[0], sizeof(double), cudaMemcpyDeviceToHost);
		return price;
	}
	__host__ double upAndOutCallPrice2(double strike,double barrier) {
		RunCallOptionPrice << <(N + blockSize - 1) / blockSize, blockSize >> >(N, d_a, strike, d_prices,barrier);
		double price;
		//cudaMemcpy(h_a, d_a, N * sizeof(MCNeuron), cudaMemcpyDeviceToHost);
		cudaMemcpy(&price, &d_prices[0], sizeof(double), cudaMemcpyDeviceToHost);
		return price;
	}
	__host__ double asianCallPrice(double strike) {
		double price = 0;
		for (int i = 0; i < N; ++i) {
			price += h_a[i].asianCallPayOff(strike);
		}
		price = price / N;
		return price;
	}
	__host__ double upAndOutCallPrice(double strike,double barrier) {
		double price = 0;
		for (int i = 0; i < N; ++i) {
			price += h_a[i].upAndOutCallPayOff(strike,barrier);
		}
		price = price / N;
		return price;
	}
	__host__ ~MCEngine() {
		delete[] h_a;
		cudaFree(d_a);
		cudaFree(d_prices);
		cudaDeviceReset();
	}
};