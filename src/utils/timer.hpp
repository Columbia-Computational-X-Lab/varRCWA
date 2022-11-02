#ifndef SPLOOSH_UTILS_TIMER
#   define SPLOOSH_UTILS_TIMER

#include <chrono>

namespace sploosh
{

inline auto now() 
{
    return std::chrono::high_resolution_clock::now();
}

template <class Clock>
inline double duration_milli_d(const std::chrono::time_point<Clock>& t1, 
                               const std::chrono::time_point<Clock>& t2)
{
    return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

}

#endif