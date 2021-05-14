#pragma once

#include <util/model.h>
#include <util/util.h>
#include <random>

NMatrixnet::TMnSseStaticMeta GenerateMeta(size_t featuresSize, size_t binaryFeaturesSize, size_t treesPerDepth) {
    // Binarization
    Y_ENSURE(binaryFeaturesSize > 0);
    Y_ENSURE(featuresSize > 0);

    float* vals = new float[binaryFeaturesSize];
    NMatrixnet::TFeature* features = new NMatrixnet::TFeature[featuresSize];

    size_t binaryFeaturesperFeature = binaryFeaturesSize / featuresSize;
    for (size_t i = 0; i < featuresSize; ++i) {
        features[i].Length = binaryFeaturesperFeature;
        features[i].Index = i;
    }
    features[0].Length = binaryFeaturesperFeature + binaryFeaturesSize % featuresSize;

    TRandomGenerator rng(179);
    for (size_t i = 0; i < binaryFeaturesSize; ++i) {
        vals[i] = rng.GenRandReal1();
    }

    NMatrixnet::TMnSseStaticMeta meta;
    meta.FeaturesSize = featuresSize;
    meta.Features = features;
    meta.ValuesSize = binaryFeaturesSize;
    meta.Values = vals;

    // Apply
    meta.SizeToCountSize = 9;
    meta.DataIndicesSize = 0;
    int* treesOnDepth = new int[meta.SizeToCountSize];
    for (size_t i = 0; i < meta.SizeToCountSize; ++i) {
        treesOnDepth[i] = treesPerDepth;
        meta.DataIndicesSize += treesPerDepth * i;
    }
    meta.SizeToCount = treesOnDepth;

    meta.Has16Bits = false;
    int* indices = new int[meta.DataIndicesSize];
    for (size_t i = 0; i < meta.DataIndicesSize; ++i) {
        reinterpret_cast<ui32*>(indices)[i] = (ui32)(rng.GenRand64() % binaryFeaturesSize) * 4;
    }
    meta.DataIndicesPtr = indices;

    return meta;
}

TVector<NMatrixnet::TMultiData::TLeafData> GenerateMultiData(size_t leafSize) {
    TRandomGenerator rng(179);
    int* data = new int[leafSize];
    for (size_t i = 0; i < leafSize; ++i) {
        reinterpret_cast<ui32*>(data)[i] = (ui32)(rng.GenRand64() % (1llu << 32));
    }
    return { NMatrixnet::TMultiData::TLeafData(data, 1, 1) };
}

NMatrixnet::TMnSseStatic GenerateModel(size_t featuresSize, size_t binaryFeaturesSize, size_t treesPerDepth) {
    NMatrixnet::TMnSseStaticMeta meta = GenerateMeta(featuresSize, binaryFeaturesSize, treesPerDepth);
    size_t leafSize = 0;
    for (size_t i = 0; i < meta.SizeToCountSize; ++i) {
        leafSize += (1 << i) * meta.SizeToCount[i];
    }
    return { meta, NMatrixnet::TMnSseStaticLeaves(NMatrixnet::TMultiData(GenerateMultiData(leafSize), leafSize)) };
}

size_t GetNumFeats(const NMatrixnet::TMnSseStatic& model) {
    size_t res = 0;
    for (size_t i = 0; i < model.Meta.FeaturesSize; ++i) {
        res = std::max<size_t>(res, model.Meta.Features[i].Index);
    }
    return res + 1;
}

void DestroyModel(NMatrixnet::TMnSseStatic& model) {
    delete[] model.Meta.Features;
    delete[] model.Meta.Values;
    delete[] model.Meta.SizeToCount;
    delete[] reinterpret_cast<const ui32*>(model.Meta.DataIndicesPtr);
    delete[] Get<NMatrixnet::TMultiData>(model.Leaves.Data).MultiData[0].Data;
}
