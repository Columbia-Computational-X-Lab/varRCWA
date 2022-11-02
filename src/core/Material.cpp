#include "Material.h"

using namespace std;

scalex Material::permittivity(scalar lambda) const
{
    if (permittivity_.index() == 0) {
        return std::get<scalex>(permittivity_);
    } else {
        return std::get< std::function<scalex(scalar)> >(permittivity_)(lambda);
    }
}