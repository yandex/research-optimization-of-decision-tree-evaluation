#pragma once

#include <util/util.h>

struct TNaiveBinarizationFormat {
    TVector<TVector<bool>> Features;
    TVector<TVector<bool>> ToVector() const {
        return Features;
    }
};
