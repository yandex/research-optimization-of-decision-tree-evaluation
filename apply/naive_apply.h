#pragma once

#include <binarization_formats/naive_format.h>
#include <util/model.h>

TVector<double> NaiveApply(const NMatrixnet::TMnSseStatic &mn, const TNaiveBinarizationFormat& binarization) {
    TVector<double> results(binarization.Features.size());
    const TVector<TMultiData::TLeafData>& multiData = Get<TMultiData>(mn.Leaves.Data).MultiData;
    Y_ENSURE(multiData.size() == 1);
    for (size_t k = 0; k < results.size(); ++k) {
        int dataIdx = 0;
        int treeCondIdx = 0;
        i64 res = 0L;
        i64 totalNumTrees = 0L;
        for (size_t condNum = 0; condNum < mn.Meta.SizeToCountSize; ++condNum) {
            const int treesNum = mn.Meta.SizeToCount[condNum];
            totalNumTrees += treesNum;
            for (int treeIdx = 0; treeIdx < treesNum; ++treeIdx) {
                int valueIdx = 0;
                for (size_t i = 0; i < condNum; ++i) {
                    Y_ENSURE((size_t)treeCondIdx < mn.Meta.DataIndicesSize);
                    int condIdx = 0;
                    if (mn.Meta.Has16Bits) {
                        condIdx = reinterpret_cast<const ui16*>(mn.Meta.DataIndicesPtr)[treeCondIdx] / 4;
                    } else {
                        condIdx = reinterpret_cast<const ui32*>(mn.Meta.DataIndicesPtr)[treeCondIdx] / 4;
                    }
                    valueIdx += static_cast<size_t>(binarization.Features[k][condIdx]) << i;
                    treeCondIdx++;
                }
                res += ui32(multiData[0].Data[dataIdx + valueIdx]);
                dataIdx += 1 << condNum;
            }
        }
        res -= totalNumTrees << 31;
        results[k] = res * multiData[0].Norm.DataScale + multiData[0].Norm.DataBias;
    }
    return results;
}
