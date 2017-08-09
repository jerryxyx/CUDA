// Xugong Li add this class for teaching
// March 21, 2011

#include "VanillaEngine.h"
#include <cmath>

VanillaEngine::VanillaEngine(const Wrapper<VanillaOption>& TheProduct_, const Parameters& r_):
                 TheProduct(TheProduct_),   r(r_),   thisPayOff (0.0)
{
	Discount = exp(-r.Integral(0,TheProduct_->GetExpiry()));
}

void VanillaEngine::DoSimulation(StatisticsMC& TheGatherer, unsigned long NumberOfPaths)
{
    double thisSpot = 0;
    
    for (unsigned long i =0; i < NumberOfPaths; ++i)
    {
        GetOnePath(thisSpot); // Use the gaussian number to simulate a spot value

        double thisPayOff = DoOnePath(thisSpot); // calculating thisPayoff by thisSpot and discount

        TheGatherer.DumpOneResult(thisPayOff); 
    }

    return;
}

// Calculate final payoff by SpotValue and discount;
// SpotValue --> PayOff, then *discount 
double VanillaEngine::DoOnePath(const double& SpotValue) const
{

	double Value= TheProduct->OptionPayOff(SpotValue);

	Value = Value*Discount;

    return Value;
}
