#pragma once
#include "defns.h"
#include <cmath>


inline scalex sinc(scalar x)
{
	return x == .0 ? scalex(1.) : scalex(sin(Pi * x) / (Pi * x));
}