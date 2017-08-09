#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
//#include <random>

__global__ void initCurandState(unsigned int seed, curandState_t *d_state) {
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed, id, 0, &d_state[id]);
}
class MCNeuron {
private:
	float spot;
	double endprice;
	float r;
	float vol;
	float timeToMaturity;
	unsigned int n_steps;
	unsigned int id;
	double max;
	double min;
	long double mean;
public:
	__device__ __host__ MCNeuron(double spot_ = 20, double r_ = 0.02, double vol_ = 0.2, double timeToMaturity_ = 1, double n_steps = 300) :
		spot(spot_),
		endprice(spot_),
		r(r_),
		vol(vol_),
		timeToMaturity(timeToMaturity_),
		n_steps(n_steps),
		min(spot_),
		max(spot_),
		mean(spot_) {};
	__device__ __host__ void resetNeuron(double spot_ = 20, double r_ = 0.02, double vol_ = 0.2, double timeToMaturity_ = 1, double n_steps_ = 300) {
		spot = spot_;
		r = r_;
		vol = vol_;
		timeToMaturity = timeToMaturity_;
		n_steps = n_steps_;
	}
	__device__ void addOne() {
		++spot;
	}
	__device__ void doOneTrail(unsigned int seed) {
		curandState_t curandState;
		curand_init(seed, id, 0, &curandState);
		double dt = timeToMaturity / n_steps;
		double Z,dW;
		for (int i = 0; i < n_steps; ++i) {
			
			//Euler Scheme for geometric brownian motion
			//Todo: Milstein scheme
			//Todo: Heston model
			Z = curand_normal(&curandState);
			endprice = endprice + endprice*r*dt + endprice*vol*sqrt(dt)*Z;
			if (endprice > max) {
				max = endprice;
			}
			else if (endprice < min) {
				min = endprice;
			}
			mean += endprice;
		}
		mean = mean / (n_steps + 1);
	}
	__device__ void doOneTrail2(curandState_t state) {
		double dt = timeToMaturity / n_steps;
		double Z;
		for (int i = 0; i < n_steps; ++i) {

			//Euler Scheme for geometric brownian motion
			//Todo: Milstein scheme
			//Todo: Heston model
			Z = curand_normal(&state);
			endprice = endprice + endprice*r*dt + endprice*vol*sqrt(dt)*Z;
			if (endprice > max) {
				max = endprice;
			}
			else if (endprice < min) {
				min = endprice;
			}
			mean += endprice;
		}
		mean = mean / (n_steps + 1);
	}
	__host__ __device__ double vanillaCallPayOff(float strike) {
		if (endprice > strike) {
			return endprice - strike;
		}
		else {
			return 0;
		}
	}
	__host__ __device__ double vanillaPutPayOff(float strike) {
		if (endprice < strike) {
			return strike - endprice;
		}
		else {
			return 0;
		}
	}
	 __host__ __device__ double asianCallPayOff(float strike) {
		 if (mean > strike) {
			 return mean - strike;
		 }
		 else {
			 return 0;
		 }
	 }
	 __host__ __device__ double asianPutPayOff(float strike) {
		 if (mean < strike) {
			 return strike - mean;
		 }
		 else {
			 return 0;
		 }
	 }
	 __host__ __device__ double upAndOutCallPayOff(float strike,float barrier) {
		 if (max > barrier || endprice<strike) {
			 return 0;
		 }
		 else {
			 return endprice-strike;
		 }
	 }
	 __host__ __device__ double upAndOutPutPayOff(float strike, float barrier) {
		 if (max > barrier || endprice>strike) {
			 return 0;
		 }
		 else {
			 return strike - strike;
		 }
	 }
	 __host__ __device__ double downAndInCallPayOff(float strike,float barrier) {
		 if (min > barrier || endprice<strike) {
			 return 0;
		 }
		 else {
			 return strike - strike;
		 }
	 }
	 __host__ __device__ double downAndInPutPayOff(float strike, float barrier) {
		 if (min > barrier || endprice>strike) {
			 return 0;
		 }
		 else {
			 return strike - strike;
		 }
	 }
	__host__  void printX() {
		std::cout << "endprice:" << endprice << "  mean:" << mean << "  max:" << max << "  min:" << min << '\n';
	}
};