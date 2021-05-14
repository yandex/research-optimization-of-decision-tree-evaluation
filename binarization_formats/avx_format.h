#pragma once

#include <util/model.h>
#include "subbatch.h"

#if defined _ubsan_enabled_
#define NFORCE_UNROLL
#else
#define NFORCE_UNROLL _Pragma("unroll")
#endif

#if defined NPROFILE
#define NINLINE_UNLESS_PROFILE __attribute__ ((noinline))
#else
#define NINLINE_UNLESS_PROFILE NFORCED_INLINE
#endif

template<size_t WordLen, bool Ordered>
class TAvxBinarization {
public:
    TAvxBinarization(const NMatrixnet::TMnSseStaticMeta &info, size_t numDocs) {
        NumDocs = numDocs;
        ValuesSize = info.ValuesSize;
        if (NumDocs > 0) {
            size_t numSubBatches = (NumDocs + WordLen - 1) / WordLen;
            const size_t valSize = info.ValuesSize * (WordLen / 8);
            Vals.template reset(new char[valSize * numSubBatches + 0x80]);
            char* valsAligned = Vals.get() + (0x40 - (uintptr_t)Vals.get() % 0x40);
            Batches.reserve(numSubBatches);
            for (size_t i = 0; i < numSubBatches - 1; ++i) {
                Batches.emplace_back(WordLen, reinterpret_cast<ui32*>(valsAligned + i * valSize));
            }
            Batches.emplace_back(NumDocs - (numSubBatches - 1) * WordLen, reinterpret_cast<ui32*>(valsAligned + (numSubBatches - 1) * valSize));
        }
    }
    size_t GetNumDocs() const {
        return NumDocs;
    }
    TVector<TVector<bool>> ToVector() const {
        TVector<TVector<bool>> res(NumDocs + WordLen, TVector<bool>(ValuesSize));
        for (size_t i = 0; i < Batches.size(); ++i) {
            for (size_t j = 0; j < ValuesSize; ++j) {
                for (size_t k = 0; k < WordLen; ++k) {
                    if constexpr (Ordered) {
                        res[i * WordLen + k][j] = (Batches[i].GetVal()[(j * WordLen + k) / 32] >> (k % 32)) & 1;
                    } else {
                        if constexpr (WordLen == 512) {
                            res[i * WordLen + (k / 8) % 4 + (k / 8) / 4 * 16 % 64 + (k / 8) / 4 * 16 / 64 * 4 + k % 8 * 64][j]
                                = (Batches[i].GetVal()[(j * WordLen + k) / 32] >> (k % 32)) & 1;
                        } else if constexpr (WordLen == 256) {
                            res[i * WordLen + (k / 8) % 4 + (k / 8) / 4 * 8 % 32 + (k / 8) / 4 * 8 / 32 * 4 + k % 8 * 32][j]
                                = (Batches[i].GetVal()[(j * WordLen + k) / 32] >> (k % 32)) & 1;
                        } else if constexpr (WordLen == 128) {
                            res[i * WordLen + (k / 8) + k % 8 * 16][j]
                                = (Batches[i].GetVal()[(j * WordLen + k) / 32] >> (k % 32)) & 1;
                        }
                    }
                }
            }
        }
        res.resize(NumDocs);
        return res;
    }

    TVector<TPreparedSubBatch> Batches;

private:
    size_t NumDocs = 0;
    size_t ValuesSize = 0;
    std::unique_ptr<char[]> Vals;
};
