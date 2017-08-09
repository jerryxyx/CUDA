#pragma once
#include "vanillaengine.h"
#include "Random2.h"

 

class VanillaBSEngine : public VanillaEngine
{
public:

      VanillaBSEngine(const Wrapper<VanillaOption>& TheProduct_,
                     const Parameters& R_,
                     const Parameters& Vol_,
                     const Wrapper<RandomBase>& TheGenerator_,
                     double Spot_);
	
      virtual void GetOnePath(double& SpotValues);
      virtual ~VanillaBSEngine(){}
 

private:
	double movedSpot;
	double rootVariance;

    Wrapper<RandomBase> TheGenerator;
    MJArray Variaterray; 
};