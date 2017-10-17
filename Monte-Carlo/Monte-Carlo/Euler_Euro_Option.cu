#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>


#include <iostream>
#include <cmath>
#include <string>
#include <chrono>


const long int n_trials = 100000;
const int blockSize = 512;

__global__ void europeanOption(
	bool isCall,
	double rate, double volatility,
	double initialPrice, double strikePrice,
	double timeToMature,
	int size, int n_iters,
	float *d_price, float *d_stockPrice,
	curandState_t *d_state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	double dt = timeToMature / double(n_iters);
	double optionValue = initialPrice;

	if (tid < size)
	{

		for (int i = 0; i < n_iters; i++)
		{
			//optionValue *= 1 + rate *dt + curand_normal(&d_state[tid])*volatility * sqrt(dt);
			//For geometric brownian motion
			//ST=St*exp( integral(t,T,r(u)-sigma(u)**2,du) + integral(t,T,sigma(u),dW(u)) )
			//Euler Scheme1: S(t+dt) = S(t)*exp( (r(t)-0.5*sigma(t)**2)*dt + sigma(t)*square(dt)*Z )
			//Euler Scheme2: S(t+dt) = S(t) + S(t)*r(t)*dt + S(t)*sigma(t)*square(dt)*Z
			//Milstein Scheme1: S(t+dt) = S(t)*exp( (r(t)-0.5*sigma(t)**2)*dt + sigma(t)*square(dt)*Z )
			//Milstein Scheme2: S(t+dt) = S(t) + S(t)*r(t)*dt + S(t)*sigma(t)*square(dt)*Z + 0.5*S(t)*sigma(t)**2 * (Z**2-1)
			//exponential
			optionValue *= exp((rate - 0.5*volatility*volatility)*dt + volatility*curand_normal(&d_state[tid]) * sqrt(dt));
			//optionValue = optionValue + optionValue*rate*dt + optionValue*volatility*sqrt(dt)*curand_normal(&d_state[tid]) + 0.5*optionValue*volatility*volatility*(curand_normal(&d_state[tid])*curand_normal(&d_state[tid]) - 1);
		}
		d_stockPrice[tid] = optionValue;//stock price at expiration
		if (isCall)
			optionValue -= strikePrice;
		else
			optionValue = strikePrice - optionValue;
		if (optionValue < 0)
			optionValue = 0;
		d_price[tid] = optionValue;//option value at expiration
	}

}

__global__ void init(
	unsigned int seed,
	curandState_t *d_state)
{
	curand_init(
		seed,
		threadIdx.x + blockDim.x * blockIdx.x,
		0,
		&d_state[threadIdx.x + blockDim.x * blockIdx.x]);
}

__global__ void init2(
	unsigned int seed1,
	unsigned int seed2,
	curandState_t *d_state1,
	curandState_t *d_state2)
{
	curand_init(
		seed1,
		threadIdx.x + blockDim.x * blockIdx.x,
		0,
		&d_state1[threadIdx.x + blockDim.x * blockIdx.x]);
	curand_init(
		seed2,
		threadIdx.x + blockDim.x * blockIdx.x,
		0,
		&d_state2[threadIdx.x + blockDim.x * blockIdx.x]);

}

int main()
{

	float *h_prices,*h_stockPrices, *d_prices, *d_stockPrices;
	typedef std::chrono::high_resolution_clock Time;
	typedef std::chrono::milliseconds ms;
	typedef std::chrono::duration<float> fsec;
	float rate, volatility, timeToMature, n_iters, stockPrice, exercisePrice;
	

	h_prices = new float[n_trials];
	h_stockPrices = new float[n_trials];
	cudaMalloc((void**)&d_prices, n_trials * sizeof(float));
	cudaMalloc((void**)&d_stockPrices, n_trials * sizeof(float));
	curandState_t *d_state;
	curandState_t *d_state1;
	curandState_t *d_state2;
	cudaMalloc((void**)&d_state, n_trials * sizeof(curandState_t));
	cudaMalloc((void**)&d_state1, n_trials * sizeof(curandState_t));
	cudaMalloc((void**)&d_state2, n_trials * sizeof(curandState_t));
	std::string optionType;
	bool isCall;
	
	std::cout << "Call option or put option? (c/p)" << std::endl;
	std::cin >> optionType;
	if (optionType == "c" || optionType == "call")isCall = true;
	else if (optionType == "p" || optionType =="put")isCall = false;
	else {
		std::cout << "wrong input!" << std::endl;
		goto Error;
	}
	//std::cout << "pleas input rate, volatility, time to mature, n_steps, stock price, exercise price:" << std::endl;
	//std::cin >> rate >> volatility >> timeToMature >> n_iters >> stockPrice >> exercisePrice;
	rate = 0.02;
	volatility = 0.2;
	timeToMature = 1;
	n_iters = 300;
	stockPrice = 20;
	exercisePrice = 20;

	auto time1 = Time::now();
	init << < (n_trials + blockSize - 1) / blockSize, blockSize >> >(time(nullptr), d_state);

	europeanOption << <(n_trials + blockSize - 1) / blockSize, blockSize >> >(
		isCall,
		rate,volatility,
		stockPrice, exercisePrice,
		timeToMature,
		n_trials, n_iters,
		d_prices, d_stockPrices,
		d_state);

	cudaMemcpy(h_prices, d_prices, n_trials * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_stockPrices, d_stockPrices, n_trials * sizeof(float), cudaMemcpyDeviceToHost);
	auto time2 = Time::now();
	fsec fs = time2 - time1;//simulation duration

	//device statics for option price
	float meanOptionValue = 0;
	float stdOptionValue = 0;
	for (int i = 0; i < n_trials; i++){
		meanOptionValue += h_prices[i];
	}
	meanOptionValue /= n_trials;
	for (int i = 0; i < n_trials; i++) {
		stdOptionValue += pow((h_prices[i] - meanOptionValue), 2);
	}
	stdOptionValue /= (n_trials - 1);
	stdOptionValue = sqrt(stdOptionValue);
	//device statics analysis for underlying asset price
	float meanStockPrice = 0;
	float stdStockPrice = 0;
	for (int i = 0; i < n_trials; i++) {
		meanStockPrice += h_stockPrices[i];
	}
	meanStockPrice /= n_trials;
	for (int i = 0; i < n_trials; i++) {
		stdStockPrice += pow((h_prices[i] - meanStockPrice), 2);
	}
	stdStockPrice /= (n_trials - 1);
	stdStockPrice = sqrt(stdStockPrice);
	//device statics analysis for log underlying asset price
	int zeroCounter = 0;
	float meanLogStockPrice = 0;
	float stdLogStockPrice = 0;
	for (int i = 0; i < n_trials; i++) {
		if(h_stockPrices[i]!=0)
			meanLogStockPrice += log(h_stockPrices[i]);
		else {
			++zeroCounter;
		}
	}
	meanLogStockPrice /= n_trials;
	for (int i = 0; i < n_trials; i++) {
		if(h_stockPrices[i]!=0)
			stdLogStockPrice += pow((log(h_stockPrices[i]) - meanLogStockPrice), 2);
	}
	stdLogStockPrice /= (n_trials - 1);
	stdLogStockPrice = sqrt(stdLogStockPrice);

	std::cout << "This is Monte-Carlo simulation using Euler method." << std::endl;
	std::cout << "rate: " << rate << "\n" << "volatility: " << volatility << "\n"<<"time to mature: "<<timeToMature<<std::endl;
	std::cout << "stock price: " << stockPrice<<"\n"<< "exercise price: " << exercisePrice << std::endl;
	std::cout << "Trials: " << n_trials << std::endl;
	std::cout << "Steps: " << n_iters << std::endl;
	std::cout << "Time comsumed: "<<fs.count() <<"s"<< std::endl;
	std::cout << "The mean of option value: " << meanOptionValue << std::endl;
	std::cout << "The std of option value: " << stdOptionValue << std::endl;
	std::cout << "The mean of underlying asset: " << meanStockPrice << std::endl;
	std::cout << "The std of of underlying asset: " << stdStockPrice << std::endl;
	std::cout << "The mean of log underlying asset: " << meanLogStockPrice << std::endl;
	std::cout << "The std of log underlying asset: " << stdLogStockPrice << std::endl;
	std::cout << "**************************************************************************************************" << std::endl;
	std::cout << "Comparation:" << std::endl;
	std::cout << "mean(log(samples of future stockPrices))=" << meanLogStockPrice << "\t" << "log(stockPrice)=" << log(stockPrice) << std::endl;
	std::cout << "std(log(samples of future stockPrices))=" << stdLogStockPrice << "\t" << "volatility*sqrt(time to mature)=" << volatility*sqrt(timeToMature) << std::endl;
	std::cout << "mean error: " << 100*(meanLogStockPrice - log(stockPrice)) / log(stockPrice) << "%"<<"\t" 
		<< "std error: " << 100*(stdLogStockPrice - volatility*sqrt(timeToMature)) / volatility*sqrt(timeToMature) <<"%"<< std::endl;
	std::cout << "The number of zero stock price: " << zeroCounter <<" Ratio: "<<100*float(zeroCounter)/n_trials<<"%"<< std::endl;

	delete[] h_prices,h_stockPrices;
	cudaFree(d_state);
	cudaFree(d_prices);
	cudaFree(d_stockPrices);
	cudaDeviceReset();
	system("pause");
	return 0;

Error:
	delete[] h_prices, h_stockPrices;
	cudaFree(d_state);
	cudaFree(d_prices);
	cudaFree(d_stockPrices);
	cudaDeviceReset();
	return 0;
}
