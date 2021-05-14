#pragma once

#include <binarization_formats/naive_format.h>
#include <util/model.h>

TNaiveBinarizationFormat NaiveBinarization(const NMatrixnet::TMnSseStaticMeta& mn, const TVector<TVector<float>>& plane) {
    TNaiveBinarizationFormat result;
    result.Features = TVector<TVector<bool>>(plane.size(), TVector<bool>(mn.ValuesSize));
    for (size_t k = 0; k < plane.size(); ++k) {
        size_t treeCondIdx = 0;
        for (size_t i = 0; i < mn.FeaturesSize; ++i) {
            for (size_t j = 0; j < mn.Features[i].Length; ++j) {
                result.Features[k][treeCondIdx] = plane[k][mn.Features[i].Index] > mn.Values[treeCondIdx] ? 1 : 0;
                treeCondIdx++;
            }
        }
    }
    return result;
}
