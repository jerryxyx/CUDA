#pragma once
#include "wrapper.h"
#include "Parameters.h"
#include "Vanilla3.h"
#include "MCStatistics.h"
#include <vector>
#include "CashFlow.h"
#include "Arrays.h"

class VanillaEngine
{
public:
 
    VanillaEngine(const Wrapper<VanillaOption>& TheProduct_,
                 const Parameters& r_);
    virtual ~VanillaEngine(void){}

public:
    void DoSimulation(StatisticsMC& TheGatherer, unsigned long NumberOfPaths);

	virtual void GetOnePath(double& SpotValues)=0; // Use the gaussian number to simulate a spot value
    double DoOnePath(const double& SpotValues) const; // calculating thisPayoff by thisSpot and discount

private:

    Wrapper<VanillaOption> TheProduct;
    Parameters r;
    double Discount;

	double thisPayOff;
};