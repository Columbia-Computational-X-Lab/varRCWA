#include "Parameterization.h"

int count(const Parameterization& para)
{
        int nPara = 0;
        for (const auto& map : para)
        {
                using std::max;
                int nValidPara = map.second.size();
                nPara = max(nPara, nValidPara);
        }
        return nPara;
}