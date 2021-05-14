#pragma once

#include <util/model.h>
#include <immintrin.h>
#include <binarization_formats/subbatch.h>

template<class X>
static X *GetAligned(X *val) {
    uintptr_t off = ((uintptr_t)val) & 0xf;
    val = (X *)((ui8 *)val - off + 0x10);
    return val;
}

class TMxNetSseBinarization {
public:
    TMxNetSseBinarization(const NMatrixnet::TMnSseStaticMeta &info, size_t numDocs) {
        NumDocs = numDocs;
        ValuesSize = info.ValuesSize;
        if (NumDocs > 0) {
            size_t numSubBatches = (NumDocs + 128 - 1) / 128;
            const size_t valSize = sizeof(ui32) * info.ValuesSize * 4;
            Vals.reset(new char[valSize * numSubBatches + 0x20]);
            char* valsAligned = GetAligned(Vals.get());
            Batches.reserve(numSubBatches);
            for (size_t i = 0; i < numSubBatches - 1; ++i) {
                Batches.emplace_back(128, reinterpret_cast<ui32*>(valsAligned + i * valSize));
            }
            Batches.emplace_back(NumDocs - (numSubBatches - 1) * 128, reinterpret_cast<ui32*>(valsAligned + (numSubBatches - 1) * valSize));
        }
    }
    size_t GetNumDocs() const {
        return NumDocs;
    }
    TVector<TVector<bool>> ToVector() const {
        TVector<TVector<bool>> res(NumDocs + 128, TVector<bool>(ValuesSize));
        for (size_t i = 0; i < Batches.size(); ++i) {
            for (size_t j = 0; j < ValuesSize; ++j) {
                for (size_t k = 0; k < 128; ++k) {
                    size_t k16 = k * 16;
                    res[i * 128 + k16 % 128 + k16 / 128][j] = (Batches[i].GetVal()[j * 4 + k / 32] >> (k % 32)) & 1;
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
