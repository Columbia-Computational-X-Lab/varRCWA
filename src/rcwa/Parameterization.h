#pragma once
#include "defns.h"
#include <unordered_map>

/**
 * Parameterization
 * It is a map from parameter to the coordinates
 * x_i = a_1 * p_1 + a_2 * p_2 + ... + a_k * p_k + x_i0
 * then the parameters are p_1, p_2, ..., p_k
 * we store a map as
 * (i, VectorXs(a_1, a_2, ..., a_k)), x_i0 is not needed
 * the max length of VectorXs is taken as k, the number of parameters
 * the default value for cofficients a_k is 0.
 */
typedef std::unordered_map<int, Eigen::VectorXs> Parameterization;

inline bool is_varied(const Parameterization& para)
{
        return para.size() > 0;
}

/**
 * Counts the number of paramter in para
 */
int count(const Parameterization& para);