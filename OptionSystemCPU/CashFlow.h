#pragma once

class CashFlow
{
public:

    double Amount;
    unsigned long TimeIndex;
    
    CashFlow(unsigned long TimeIndex_=0UL, double Amount_=0.0) 
                : TimeIndex(TimeIndex_),
                  Amount(Amount_){};
 
};