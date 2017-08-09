#include "VanillaBSEngine.h"
#include <cmath>

// Use the gaussian number to simulate a spot value
void VanillaBSEngine::GetOnePath(double& SpotValues) 
{
    TheGenerator->GetGaussians(Variaterray);
	SpotValues = movedSpot * exp (rootVariance*Variaterray[0]);

	return;
}

VanillaBSEngine::VanillaBSEngine(const Wrapper<VanillaOption>& TheProduct_,
                                    const Parameters& R_,
                                    const Parameters& Vol_,
                                    const Wrapper<RandomBase>& TheGenerator_,
                                    double Spot_)
                                    :
                                    VanillaEngine(TheProduct_,R_),
                                    TheGenerator(TheGenerator_)//	,Variaterray(1)
{
 
    TheGenerator->ResetDimensionality(1);

	double Expiry = TheProduct_->GetExpiry();
	double variance = Vol_.IntegralSquare(0,Expiry);
	double itoCorrection = -0.5*variance;
	
	rootVariance = sqrt(variance);	
	movedSpot    = Spot_*exp(R_.Integral(0,Expiry) + itoCorrection);

	Variaterray.resize(1);
}

