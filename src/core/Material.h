#pragma once
#include "rcwa/defns.h"
#include <variant>

class Material 
{
private:
    std::variant< scalex, std::function<scalex(scalar)> > permittivity_;

public:
    Material(scalex permittivity): permittivity_(permittivity) { }
    Material(std::function<scalex(scalar)> permittivity):
        permittivity_(permittivity) {}
    scalex permittivity(scalar lambda) const;
};

const Material m_SiO2 = {1.445 * 1.445};
const Material m_Si = {3.48 * 3.48};