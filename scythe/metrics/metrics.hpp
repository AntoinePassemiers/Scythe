#ifndef METRICS_HPP_
#define METRICS_HPP_

#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../id3.hpp"

/*
Reference
---------

http://luthuli.cs.uiuc.edu/~daf/courses/optimization/papers/2699986.pdf
pg 1201
*/

namespace gbdf {
    // Classification
    constexpr int MLOG_LOSS = 0x7711A0;

    // Regression
    constexpr int MSE = 0xC97B00; // Mean squared error
}

typedef double loss_t;

#endif // METRICS_HPP_