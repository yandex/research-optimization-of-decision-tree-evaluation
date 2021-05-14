#pragma once

#include "generate_model.h"

#include <benchmark/benchmark.h>
#include <cstdlib>

struct TModelAndDataHolder {
    TModelAndDataHolder(size_t docCount, size_t featuresSize, size_t binaryFeaturesSize, size_t treesPerDepth=0) {
        Model = GenerateModel(featuresSize, binaryFeaturesSize, treesPerDepth);
        GenerateData(docCount);
    }

    void GenerateData(size_t docCount) {
        TRandomGenerator rng(179);
        auto fCount = GetNumFeats(Model);

        Features.resize(docCount);
        for (size_t i = 0; i < docCount; ++i) {
            Features[i].resize(fCount);
            for (size_t j = 0; j < fCount; ++j) {
                Features[i][j] = rng.GenRandReal1();
            }
        }
    }

    ~TModelAndDataHolder() {
        DestroyModel(Model);
    }

    NMatrixnet::TMnSseStatic Model;
    TVector<TVector<float>> Features;
};

template<typename TResult>
class TResultComparer {
public:
    void AddResult(const TString& methodName, const TString& modelName, size_t docCount, TResult result) {
        Results[{modelName, docCount}][methodName] = result;
    }
    ~TResultComparer() {
        for (const auto& [input, methods] : Results) {
            for (auto i = methods.begin(); i != methods.end(); ++i) {
                for (auto j = std::next(i); j != methods.end(); ++j) {
                    if constexpr (std::is_same_v<TResult, TVector<double>>) {
                        size_t k = 0;
                        while (k < i->second.size() && std::abs(i->second[k] - j->second[k]) == 0) {
                            ++k;
                        }
                        if (k < i->second.size()) {
                            Cerr << "\033[1;31mResults different between " << input.first << "Model" << i->first << "/" << input.second
                                << " and " << input.first << "Model" << j->first << "/" << input.second << "\033[0m\n";
                            Cerr << "Line " << k << ": " << i->second[k] << " " << j->second[k] << "\n";
                        }
                    } else if constexpr (std::is_same_v<TResult, TVector<TVector<bool>>>) {
                        if (i->second != j->second) {
                            Cerr << "\033[1;31mResults different between " << input.first << "Model" << i->first << "/" << input.second
                                                << " and " << input.first << "Model" << j->first << "/" << input.second << "\033[0m\n";

                            size_t k = 0;
                            while (i->second[k] == j->second[k]) {
                                ++k;
                            }
                            Cerr << "Line " << k << "\n";
                        }
                    }
                }
            }
        }
    }
private:
    TMap<std::pair<TString, size_t>, TMap<TString, TResult>> Results;
};

TResultComparer<TVector<TVector<bool>>> BinarizationResultsComparer;


#define BENCH_SYNTHETIC_BINARIZATION(FeaturesSize, BinaryFeaturesSize, BinarizationFunction)                                      \
void DoBinarization##FeaturesSize##_##BinaryFeaturesSize##BinarizationFunction(size_t docCount, benchmark::State& state) {        \
    TModelAndDataHolder modelAndData(docCount, FeaturesSize, BinaryFeaturesSize);                                                 \
    for (auto _ : state) {                                                                                                        \
        benchmark::DoNotOptimize(BinarizationFunction(modelAndData.Model.Meta, modelAndData.Features));                           \
    }                                                                                                                             \
    if (state.thread_index == 0) {                                                                                                \
        BinarizationResultsComparer.AddResult(#BinarizationFunction, "FS" #FeaturesSize "_BFS" #BinaryFeaturesSize, docCount,     \
            BinarizationFunction(modelAndData.Model.Meta, modelAndData.Features).ToVector());                                     \
    }                                                                                                                             \
}                                                                                                                                 \
void FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction(benchmark::State& state) {                               \
    DoBinarization##FeaturesSize##_##BinaryFeaturesSize##BinarizationFunction(state.range(0), state);                             \
}                                                                                                                                 \
BENCHMARK(FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction)->DenseRange(1, 10, 1);                           \
BENCHMARK(FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction)->Arg(1024)->Threads(4)->Threads(8);


TVector<float*> Transpose(const NMatrixnet::TMnSseStaticMeta& info, const TVector<TVector<float>> &features, void*& buffer) {
    size_t fsize = sizeof(float) * (((features.size() - 1) / 512 + 1) * 512);
    buffer = malloc(fsize * info.FeaturesSize + 0x80);
    memset(buffer, 0, fsize * info.FeaturesSize + 0x80);
    float* start = reinterpret_cast<float*>((char*)buffer + (0x40 - (uintptr_t)buffer % 0x40));
    TVector<float*> res(info.FeaturesSize);
    for (size_t i = 0; i < info.FeaturesSize; ++i) {
        res[i] = start + i * fsize / sizeof(float);
        for (size_t j = 0; j < features.size(); ++j) {
            res[i][j] = features[j][info.Features[i].Index];
        }
    }
    return res;
}

#define BENCH_SYNTHETIC_BINARIZATION_TRANSPOSED(FeaturesSize, BinaryFeaturesSize, BinarizationFunction)                           \
void DoBinarization##FeaturesSize##_##BinaryFeaturesSize##BinarizationFunction(size_t docCount, benchmark::State& state) {        \
    TModelAndDataHolder modelAndData(docCount, FeaturesSize, BinaryFeaturesSize);                                                 \
    void* holder = nullptr;                                                                                                       \
    auto transposed = Transpose(modelAndData.Model.Meta, modelAndData.Features, holder);                                          \
    for (auto _ : state) {                                                                                                        \
        benchmark::DoNotOptimize(BinarizationFunction(modelAndData.Model.Meta, transposed, modelAndData.Features.size()));        \
    }                                                                                                                             \
    if (state.thread_index == 0) {                                                                                                \
        BinarizationResultsComparer.AddResult(#BinarizationFunction, "FS" #FeaturesSize "_BFS" #BinaryFeaturesSize, docCount,     \
            BinarizationFunction(modelAndData.Model.Meta, transposed, modelAndData.Features.size()).ToVector());                  \
    }                                                                                                                             \
    free(holder);                                                                                                                 \
}                                                                                                                                 \
void FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction(benchmark::State& state) {                               \
    DoBinarization##FeaturesSize##_##BinaryFeaturesSize##BinarizationFunction(state.range(0), state);                             \
}                                                                                                                                 \
BENCHMARK(FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction)->DenseRange(1, 1024, 1);                           \
BENCHMARK(FS##FeaturesSize##_BFS##BinaryFeaturesSize##_##BinarizationFunction)->Arg(1024)->Threads(4)->Threads(8);


TResultComparer<TVector<double>> ApplyResultComparer;

void AlignModel(NMatrixnet::TMnSseStatic &mn) {
    TVector<NMatrixnet::TMultiData::TLeafData>& multiData = Get<NMatrixnet::TMultiData>(mn.Leaves.Data).MultiData;
    size_t wc512 = 0;
    for (size_t i = 0; i < mn.Meta.SizeToCountSize; ++i) {
        wc512 += ((1 << i) * 32 + 511) / 512 * mn.Meta.SizeToCount[i];
    }
    const int* src = multiData[0].Data;
    char* buffer = (char*)calloc(0x80 + wc512 * 512 / 8, 1);
    int* dst = (int*)(buffer + (0x40 - (uintptr_t)buffer % 0x40));
    for (size_t i = 0; i < mn.Meta.SizeToCountSize; ++i) {
        if (i < 4) {
            for (int j = 0; j < mn.Meta.SizeToCount[i]; ++j) {
                memcpy(dst, src, (1 << i) * sizeof(int));
                dst += 512 / 32;
                src += (1 << i);
            }
        } else {
            memcpy(dst, src, (1 << i) * sizeof(int) * mn.Meta.SizeToCount[i]);
            dst += (1 << i) * mn.Meta.SizeToCount[i];
            src += (1 << i) * mn.Meta.SizeToCount[i];
        }
    }
    multiData[0].Data = (int*)(buffer);
}

#define BENCH_SYNTHETIC_APPLY(BinaryFeaturesSize, TreesPerDepth, BinarizationFunction, ApplyFunction)              \
void DoApply##BinaryFeaturesSize_##TreesPerDepth##ApplyFunction(size_t docCount, benchmark::State& state) {        \
    TModelAndDataHolder modelAndData(docCount, BinaryFeaturesSize, BinaryFeaturesSize, TreesPerDepth);             \
    auto orig = Get<NMatrixnet::TMultiData>(modelAndData.Model.Leaves.Data).MultiData[0].Data;                     \
    if (TString(#ApplyFunction).find("ShuffleFetch") != TString::npos) {                                           \
        AlignModel(modelAndData.Model);                                                                            \
    }                                                                                                              \
    auto binarization = BinarizationFunction(modelAndData.Model.Meta, modelAndData.Features);                      \
    for (auto _ : state) {                                                                                         \
        benchmark::DoNotOptimize(ApplyFunction(modelAndData.Model, binarization));                                 \
    }                                                                                                              \
    if (state.thread_index == 0) {                                                                                 \
        ApplyResultComparer.AddResult(#ApplyFunction, "BFS" #BinaryFeaturesSize "_TPD" #TreesPerDepth, docCount,   \
                                      ApplyFunction(modelAndData.Model, binarization));                            \
    }                                                                                                              \
    if (TString(#ApplyFunction).find("ShuffleFetch") != TString::npos) {                                           \
        free((void*)Get<NMatrixnet::TMultiData>(modelAndData.Model.Leaves.Data).MultiData[0].Data);                \
        Get<NMatrixnet::TMultiData>(modelAndData.Model.Leaves.Data).MultiData[0].Data = orig;                      \
    }                                                                                                              \
}                                                                                                                  \
void BFS##BinaryFeaturesSize##_TPD##TreesPerDepth##_##ApplyFunction(benchmark::State& state) {                     \
    DoApply##BinaryFeaturesSize_##TreesPerDepth##ApplyFunction(state.range(), state);                              \
}                                                                                                                  \
BENCHMARK(BFS##BinaryFeaturesSize##_TPD##TreesPerDepth##_##ApplyFunction)->DenseRange(1, 1024, 1);                 \
BENCHMARK(BFS##BinaryFeaturesSize##_TPD##TreesPerDepth##_##ApplyFunction)->Arg(1024)->Threads(4)->Threads(8);

